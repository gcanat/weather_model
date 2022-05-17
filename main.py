"""
Script to train models on our homemade 'MSBGD' image dataset.
This version uses Convolution and/or AvgPooling after base_model last layer.
We also split data in train and validation before setting up the data generators,
to ensure better balance in target classes.
See --help for information about argument options.
"""
import datetime
import os
from options import create_argparser
from config_params import POSSIBLE_LOSSES, POSSIBLE_BASE_MODELS
from load_data import load_data, make_data_generator
from model import create_model, fit_model
from metrics import eval_model, build_summary


if __name__ == "__main__":
    # parsing arguments
    parser = create_argparser(POSSIBLE_LOSSES, POSSIBLE_BASE_MODELS)
    args = parser.parse_args()
    PROJECT_PATH = args.project_path
    IMG_BASE_PATH = os.path.join(PROJECT_PATH, args.image_path)
    LABELS_DF_PATH = os.path.join(PROJECT_PATH, args.label_path)
    BASE_MODEL = args.base_model
    FREEZE_CORE = True if args.freeze_core == "True" else False
    PRETRAINED = True if args.use_pretrained == "True" else False
    LOSS = args.loss
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    AUGMENT_FACTOR = args.augment_factor
    ENABLE_CLASS_WEIGHTS = True if args.class_weight == "True" else False
    TARGET = args.target
    RANDOM_STATE = args.random_state
    WARMUP = args.warmup
    INIT_LR = args.init_lr
    DROPOUT = args.dropout

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # load data
    train_df1, train_df2 = load_data(LABELS_DF_PATH)

    if TARGET == "All":
        target_cols = list(train_df1.columns[1:])
    else:
        target_cols = [col for col in train_df1.columns if TARGET in col]

    print(target_cols)

    # create dataloaders
    train_gen, val_gen, class_cw = make_data_generator(
        train_df1, train_df2, target_cols, IMG_BASE_PATH, BATCH_SIZE, RANDOM_STATE
    )
    class_counts = class_cw[0]

    if ENABLE_CLASS_WEIGHTS:
        class_weight = class_cw[1]
    else:
        class_weight = None

    # load model
    model = create_model(
        BASE_MODEL,
        target_cols,
        class_counts,
        FREEZE_CORE,
        PRETRAINED,
        LOSS,
        curr_time,
        INIT_LR,
        DROPOUT,
    )

    # if WARMUP > 0:
    #     print("[INFO] Starting warm up stage")
    # FIXME: add learning rate callback in fit function so we can change
    # learning rate

    # train model
    history = fit_model(
        model,
        train_gen,
        val_gen,
        AUGMENT_FACTOR,
        NUM_EPOCHS,
        class_weight,
        curr_time,
        PROJECT_PATH,
        BASE_MODEL,
        INIT_LR,
        WARMUP,
    )

    # eval model
    classif_rep = eval_model(model, val_gen, target_cols)

    # create summary and write it to file
    build_summary(BASE_MODEL, PROJECT_PATH, args, classif_rep, curr_time)
