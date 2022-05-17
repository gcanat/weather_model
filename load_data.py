import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
import numpy as np

def split_data(df, y_col, random_state):
    """Split dataframe into train and validation base on target columns"""
    df_train, df_val = None, None
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    # convert labels to string for concatenation
    y_data = df[y_col].astype(int).astype(str)
    # concatenate labels into one column so we can use StratifiedKFold
    # this method is the fastest
    y_data = pd.Series(map("_".join, y_data.values.tolist()), index=y_data.index)
    # define columns of X (not really needed..)
    x_col = list(set(df.columns) - set(y_col))
    # only get the first split to get a validation set with size 0.25
    for train_ix, val_ix in skf.split(df[x_col], y_data):
        df_train, df_val = df.iloc[train_ix], df.iloc[val_ix]
        break
    print(df_train.shape, df_val.shape)
    return df_train, df_val


# ---------------- DATA CLEANING AND LOADING -------------------#
def load_data(LABELS_DF_PATH):
    """load the dataframe containing all the metadata of images for train"""
    df = pd.read_csv(
        LABELS_DF_PATH,
        index_col=0,
        dtype={
            "Image": str,
            "Timing": float,
            "Fog": float,
            "Rain": float,
            "Snow": float,
            "Illumination": float,
        },
    )

    # fixing some inconsistency in labeling
    # If image was originally tagged 'rain' then it cannot be tagged as Snow
    df.loc[df["Image"].str.contains("rain"), "Snow"] = 0
    # If image was originally tagged 'snow' then it cannot be tagged as Rain
    df.loc[df["Image"].str.contains("snow"), "Rain"] = 0

    # fill missing values with 0 for all columns except Timing
    df = df.fillna(value={"Fog": 0, "Rain": 0, "Snow": 0, "Illumination": 0}).dropna()
    # make sure all values are converted to int (0.0 to 0, etc)
    df[df.columns[1:]] = df[df.columns[1:]].astype(int)

    # compute number of classes
    target_ohe = pd.get_dummies(
        df.drop("Image", axis=1).astype(str)
    )  # need to convert to str before one-hot enc
    # NUM_CLASSES = len(target_ohe.columns)
    # make new df to use for DataGenerator
    train_df = pd.concat([df["Image"], target_ohe], axis=1)

    train_df1 = train_df.loc[
        (train_df["Timing_1"] == 0)
        & (train_df["Snow_1"] == 0)
        & (train_df["Illumination_1"] == 0),
        :,
    ]

    train_df2 = train_df.loc[
        (train_df["Timing_1"] == 1)
        | (train_df["Snow_1"] == 1)
        | (train_df["Illumination_1"] == 1),
        :,
    ]

    return train_df1, train_df2


def make_data_generator(
    train_df1, train_df2, y_col, IMG_BASE_PATH, BATCH_SIZE, RANDOM_STATE
):
    """create data generators from train sets and validation sets"""
    train1, val1 = split_data(train_df1, y_col, random_state=RANDOM_STATE)
    train2, val2 = split_data(train_df2, y_col, random_state=RANDOM_STATE)

    target_ohe = train_df1.iloc[:, 1:]
    NUM_CLASSES = len(target_ohe.columns)
    # compute class weights
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    class_counts = dict(np.mean(target_ohe, axis=0))
    print(class_counts)
    # we use the inverse proportion as weight, the formula involves NUM_CLASSES as explained in sklearn compute_class_weights
    # this heuristic is inspired by `Logistic Regression in Rare Events Data`, King, Zen, 2001.
    class_weight = {}
    for i, key in enumerate(class_counts.keys()):
        class_weight[i] = 1 / (class_counts[key] * NUM_CLASSES)

    # setting up the data generators
    # fist generator without augmentation
    train_datagen1 = ImageDataGenerator()
    train_gen1 = train_datagen1.flow_from_dataframe(
        dataframe=train1,
        directory=IMG_BASE_PATH,
        x_col="Image",
        y_col=y_col,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="raw",
    )

    val_datagen1 = ImageDataGenerator()
    val_gen1 = val_datagen1.flow_from_dataframe(
        dataframe=val1,
        directory=IMG_BASE_PATH,
        x_col="Image",
        y_col=y_col,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="raw",
    )

    # second generator with augmentation
    train_datagen2 = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.001,  # diagonal stretching
        fill_mode="reflect",
        horizontal_flip=True,
    )

    train_gen2 = train_datagen2.flow_from_dataframe(
        dataframe=train2,
        directory=IMG_BASE_PATH,
        x_col="Image",
        y_col=y_col,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="raw",
    )

    # no need for augmentation in validation set
    val_datagen2 = ImageDataGenerator()
    val_gen2 = val_datagen2.flow_from_dataframe(
        dataframe=val2,
        directory=IMG_BASE_PATH,
        x_col="Image",
        y_col=y_col,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="raw",
    )
    return (
        (train_gen1, train_gen2),
        (val_gen1, val_gen2),
        (class_counts, class_weight),
    )


def myGenerator(seq1, seq2, multiplySeq2By):
    """custom generator to yield batches coming from both generators"""
    generators = [seq1, seq2]
    # creating indices to get data from the generators
    len1 = len(seq1)
    len2 = len(seq2)
    indices1 = np.zeros((len1, 2), dtype=int)
    indices2 = np.ones((len2, 2), dtype=int)
    indices1[:, 1] = np.arange(len1)  # pairs like [0,0], [0,1], [0,2]....
    indices2[:, 1] = np.arange(len2)  # pairs like [1,0], [1,1], [1,2]....
    indices2 = [indices2] * multiplySeq2By  # repeat indices2 to generate more from it
    allIndices = np.concatenate([indices1] + indices2, axis=0)
    # randomize the order here
    np.random.shuffle(allIndices)
    # now we loop the indices infinitely to get data from the original generators
    while True:
        for g, el in allIndices:
            x, y = generators[g][el]
            yield x, y  # when training, or "yield x" when testing
        # you may want another round of shuffling here for the next epoch.
