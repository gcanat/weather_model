import os
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report


def f1(y_true, y_pred, avg="micro"):
    """
    F1 score calculated with different averaging options (as in sklearn).
      - macro: Calculate metrics for each label, and find their unweighted mean
      - micro: Calculate metrics globally by counting the total true positives,
               false negatives and false positives.
      - weighted: Calculate metrics for each label, and find their average weighted
                  by support (the number of true instances for each label).
    """
    # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    if avg == "macro":
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return K.mean(f1)
    if avg == "micro":
        precision = K.sum(true_positives) / (K.sum(predicted_positives) + K.epsilon())
        recall = K.sum(true_positives) / (K.sum(possible_positives) + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1
    if avg == "weighted":
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        ground_positives = K.sum(y_true, axis=0)
        weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
        weighted_f1 = K.sum(weighted_f1)
        return weighted_f1


def f1_mac(y_true, y_pred):
    return f1(y_true, y_pred, avg="macro")


def f1_mic(y_true, y_pred):
    return f1(y_true, y_pred, avg="micro")


def f1_loss(y_true, y_pred):
    """Custom F1 loss"""
    # we add ground_positive to calculate the weighted version
    # ground_positives = K.sum(y_true, axis=0)
    tp = K.sum(K.cast(y_true, "float") * y_pred, axis=0)
    tn = K.sum(K.cast((1 - y_true), "float") * (1 - y_pred), axis=0)
    fp = K.sum(K.cast((1 - y_true), "float") * y_pred, axis=0)
    fn = K.sum(K.cast(y_true, "float") * (1 - y_pred), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)




class F1History(tf.keras.callbacks.Callback):
    """Callback to calculate F1 score on whole dataset (train and val) after each epoch"""

    def __init__(self, train, validation=None, average="macro"):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train
        if average == "macro":
            self.f1_func = f1_mac
            self.metric_name = "f1_mac"
        elif average == "micro":
            self.f1_func = f1_mic
            self.metric_name = "f1_mic"
        else:
            raise NotImplementedError

    def on_epoch_end(self, epoch, logs={}):
        logs[self.metric_name] = float("-inf")
        y_train, y_pred = make_preds(self.model, self.train)
        score = self.f1_func(y_train.astype(float), y_pred.astype(float))

        if self.validation:
            logs[f"val_{self.metric_name}"] = float("-inf")
            y_valid, y_val_pred = make_preds(self.model, self.validation)
            val_score = self.f1_func(y_valid.astype(float), y_val_pred.astype(float))
            logs[self.metric_name] = score
            logs[f"val_{self.metric_name}"] = val_score
        else:
            logs["f1_mac"] = score


def multi_category_focal_loss(gamma=2.0, alpha=0.25):
    """
    focal loss for multi category of multi label problem
    Focal loss for multi-class or multi-label problems
         Alpha controls the weight when the true value y_true is 1/0
                 The weight of 1 is alpha, and the weight of 0 is 1-alpha.
         When your model is under-fitting and you have difficulty learning, you can try to apply this function as a loss.
         When the model is too aggressive (whenever it tends to predict 1), try to reduce the alpha
         When the model is too inert (whenever it always tends to predict 0, or a fixed constant, it means that no valid features are learned)
                 Try to increase the alpha and encourage the model to predict 1.
    Usage:
     model.compile(loss=[multi_category_focal_loss(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.0e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.math.multiply(y_true, y_pred) + tf.math.multiply(
            1 - y_true, 1 - y_pred
        )
        ce = -tf.math.log(y_t)
        weight = tf.math.pow(tf.math.subtract(1.0, y_t), gamma)
        fl = tf.math.multiply(tf.math.multiply(weight, ce), alpha_t)
        loss = tf.math.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


# ----------------------------- MODEL EVALUATION ----------------------------- #
def make_preds(model, gen):
    """Make predictions on generators"""
    generator1, generator2 = gen
    generator1.reset()
    generator2.reset()
    y_proba1 = model.predict(generator1, verbose=0)
    y_proba2 = model.predict(generator2, verbose=0)

    if isinstance(y_proba1, list):
        y_proba = np.vstack((np.hstack(y_proba1), np.hstack(y_proba2)))
    else:
        y_proba = np.vstack((y_proba1, y_proba2))

    y_pred = (y_proba > 0.5) * 1
    y_true = np.vstack((generator1.labels, generator2.labels))
    return y_true, y_pred


def eval_model(model, val_gen, labels):
    """Classification report after training"""
    y_val, y_pred = make_preds(model, val_gen)
    print("Classification report")
    classif_rep = classification_report(y_val, y_pred, target_names=labels)
    print(classif_rep)
    return classif_rep


def build_summary(base_model, project_path, args, classif_rep, curr_time):
    """Write training summary to a report.log file"""
    summary = ""
    for arg in vars(args):
        summary += f"{arg}: {getattr(args, arg)}\n"
    summary += f"\nClassification report:\n{classif_rep}"

    filename = os.path.join(project_path, f"{curr_time}_{base_model}_report.log")
    with open(filename, "a") as f:
        f.write(summary)
