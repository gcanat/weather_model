import pandas as pd
import tensorflow as tf

from focal_loss import BinaryFocalLoss
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from load_data import myGenerator
from metrics import f1_mic, f1_mac, f1_loss, F1History



# ---------------------- MODEL CREATION -------------------------#
def create_model(
    base_model,
    target_cols,
    class_counts,
    freeze_core,
    use_pretrained,
    loss,
    curr_time,
    init_lr,
    dropout,
):
    """
    Create model based on base_model and class_counts for weight initialization
    """
    # help the network to start learning by initializing bias correctly
    if "Timing_0" in target_cols:
        time_bias = tf.keras.initializers.Constant(
            [
                class_counts["Timing_0"],
                class_counts["Timing_1"],
                class_counts["Timing_2"],
                class_counts["Timing_3"],
            ]
        )
    if "Fog_0" in target_cols:
        fog_bias = tf.keras.initializers.Constant(
            [class_counts["Fog_0"], class_counts["Fog_1"], class_counts["Fog_2"]]
        )
    if "Rain_0" in target_cols:
        rain_bias = tf.keras.initializers.Constant(
            [class_counts["Rain_0"], class_counts["Rain_1"]]
        )
    if "Snow_0" in target_cols:
        snow_bias = tf.keras.initializers.Constant(
            [class_counts["Snow_0"], class_counts["Snow_1"]]
        )
    if "Illumination_0" in target_cols:
        illu_bias = tf.keras.initializers.Constant(
            [class_counts["Illumination_0"], class_counts["Illumination_1"]]
        )

    # default size of datagenerator is 256x256
    input_layer = Input(shape=[256, 256, 3])
    weights = "imagenet" if use_pretrained else None
    if base_model == "resnet50":
        # convert the input images from RGB to BGR, then will zero-center each color
        # channel with respect to the ImageNet dataset, without scaling
        x = tf.keras.applications.resnet50.preprocess_input(input_layer)
        core = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights="imagenet"
        )
    elif base_model == "vgg16":
        # convert the input images from RGB to BGR, then will zero-center each color
        # channel with respect to the ImageNet dataset, without scaling
        x = tf.keras.applications.vgg16.preprocess_input(input_layer)
        core = tf.keras.applications.VGG16(include_top=False, weights=weights)
    elif base_model == "densenet":
        # The input pixels values are scaled between 0 and 1 and each channel is
        # normalized with respect to the ImageNet dataset.
        x = tf.keras.applications.densenet.preprocess_input(input_layer)
        core = tf.keras.applications.DenseNet121(include_top=False, weights=weights)
    elif base_model == "xception":
        # scale input between -1 and 1
        x = tf.keras.applications.xception.preprocess_input(input_layer)
        core = tf.keras.applications.Xception(include_top=False, weights=weights)
    elif base_model == "effinetv2":
        # preprocessign included in model
        core = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False, weights=weights, include_preprocessing=True
        )
    # shape of output of base_model
    # resnet shape: (None, 8, 8, 2048)
    # vgg16: (None, 8, 8, 512)
    # densenet: (None, 8, 8, 1024)
    # xception: (None, 8, 8, 2048)
    conv = tf.keras.layers.Conv2D(512, (3, 3), activation="relu")
    x = conv(core(x))
    gavgpool = tf.keras.layers.GlobalAveragePooling2D()
    x = gavgpool(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.2)(x)
    output_count = 0
    outputs = []
    if "Timing_0" in target_cols:
        time_output = Dense(
            4, activation="softmax", bias_initializer=time_bias, name="time"
        )(x)
        outputs.append(time_output)
        output_count += 4
    if "Fog_0" in target_cols:
        fog_output = Dense(
            3, activation="softmax", bias_initializer=fog_bias, name="fog"
        )(x)
        outputs.append(fog_output)
        output_count += 3
    if "Rain_0" in target_cols:
        rain_output = Dense(
            2, activation="softmax", bias_initializer=rain_bias, name="rain"
        )(x)
        outputs.append(rain_output)
        output_count += 2
    if "Snow_0" in target_cols:
        snow_output = Dense(
            2, activation="softmax", bias_initializer=snow_bias, name="snow"
        )(x)
        outputs.append(snow_output)
        output_count += 2
    if "Illumination_0" in target_cols:
        illu_output = Dense(
            2, activation="softmax", bias_initializer=illu_bias, name="illu"
        )(x)
        outputs.append(illu_output)
        output_count += 2

    # freeze weights of ResNet50 (trainable = False) or not (trainable = True)
    if freeze_core:
        core.trainable = False
    else:
        core.trainable = True

    model = Model(
        inputs=input_layer,
        outputs=outputs,
    )

    # print the model specs
    print(model.summary())

    # define model metrics

    METRICS = [
        # CategoricalCrossentropy(name="crossent"),
        # CategoricalAccuracy(name="accuracy"),
        # Precision(name="precision"),
        # Recall(name="recall"),
        # f1_mac,
        f1_mic,
        f1_mac,
        # tfa.metrics.F1Score(4, average="macro"),
        # tfa.metrics.MultiLabelConfusionMatrix(4)
    ]

    # compile and train
    if loss == "focal":
        # losses = [ multi_category_focal_loss(gamma=2.0, alpha=.5)] * len(outputs)
        # losses = [tfa.losses.SigmoidFocalCrossEntropy()] * len(outputs)
        losses = [BinaryFocalLoss(gamma=2)] * len(outputs)
    elif loss == "crossent":
        losses = [tf.keras.losses.CategoricalCrossentropy()] * len(outputs)
    elif loss == "f1":
        losses = [f1_loss] * len(outputs)
    else:
        raise NotImplementedError

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=init_lr),
        loss=losses,
        metrics=METRICS,
    )
    return model


# -------------------------- MODEL TRAINING ----------------------- #
def fit_model(
    model,
    train_gen,
    val_gen,
    augment_factor,
    num_epochs,
    class_weight,
    curr_time,
    project_path,
    base_model,
    init_lr=0.01,
    warmup=0,
):
    """Training the model"""
    train_generator1, train_generator2 = train_gen
    val_generator1, val_generator2 = val_gen

    STEP_SIZE_TRAIN = len(train_generator1) + len(train_generator2) * augment_factor
    STEP_SIZE_VALID = len(val_generator1) + len(val_generator2)

    # define callbacks
    csv_logger = tf.keras.callbacks.CSVLogger(
        project_path + f"/{curr_time}_{base_model}_train_log.csv"
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )

    initial_learning_rate = init_lr
    final_learning_rate = 1e-5
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (
        1 / (num_epochs + warmup)
    )

    def scheduler(epoch, lr):
        if epoch < warmup:
            return lr
        else:
            return lr * learning_rate_decay_factor

    lr_sched = tf.keras.callbacks.LearningRateScheduler(scheduler)

    f1_score = F1History(
        train=(train_generator1, train_generator2),
        validation=(val_generator1, val_generator2),
        average="macro",
    )

    all_callbacks = [csv_logger, early_stop, f1_score, lr_sched]

    def generator_wrapper(generator):
        for batch_x, batch_y in generator:
            yield (
                batch_x,
                [
                    batch_y[:, :4],
                    batch_y[:, 4:7],
                    batch_y[:, 7:9],
                    batch_y[:, 9:11],
                    batch_y[:, 11:],
                ],
            )

    history = model.fit(
        generator_wrapper(
            myGenerator(train_generator1, train_generator2, augment_factor)
        ),
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=generator_wrapper(
            myGenerator(val_generator1, val_generator2, 1)
        ),
        validation_steps=STEP_SIZE_VALID,
        epochs=num_epochs,
        class_weight=class_weight,
        callbacks=all_callbacks,
    )

    # save training history to csv
    hist_df = pd.DataFrame(history.history)
    hist_file = f"/{curr_time}_{base_model}_history.csv"
    with open(project_path + hist_file, "w") as f:
        hist_df.to_csv(f)

    # saving model weights
    weights_final = project_path + f"/{curr_time}_{base_model}_weights.h5"
    model.save_weights(weights_final)

    return history
