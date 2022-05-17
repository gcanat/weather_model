"""
Create all the arguments of the scripts
"""
import argparse

# setting up argument parser
def create_argparser(possible_losses, possible_base_models):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_path",
        "-pp",
        default="/home/infres/canat-20/Fil_rouge",
        help="Base path for the data, including images, labels, etc.",
    )
    parser.add_argument(
        "--image_path",
        "-ip",
        default="msbgd_dataset/",
        help="relative path for image files (relative to project_path).",
    )
    parser.add_argument(
        "--label_path",
        "-lp",
        default="msbgd_dataset/00_labels/labels.csv",
        help="relative path for label file (relative to project_path).",
    )
    parser.add_argument(
        "--base_model",
        "-bm",
        default="resnet50",
        choices=possible_base_models,
        help=f"Base model to use. Can be one of : {possible_base_models}.",
    )
    parser.add_argument(
        "--freeze_core",
        "-fc",
        default="False",
        choices=["True", "False"],
        help="Wether to freeze base_model weights or not.",
    )
    parser.add_argument(
        "--use_pretrained",
        "-upt",
        default="True",
        choices=["True", "False"],
        help="Wether to load pre-trained weights (from imagenet) or not.",
    )
    parser.add_argument(
        "--loss",
        "-lo",
        default="crossent",
        choices=possible_losses,
        help=f"Type of loss to use. Can be one of: {possible_losses}.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=64,
        help="Size of batches used by DataGenerators.",
    )
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        default=30,
        help="Number of epochs used in training.",
    )
    parser.add_argument(
        "--augment_factor",
        "-af",
        type=int,
        default=5,
        help="Augmentation factor for under-represented classes, namely Timing_1, Snow_1, Illumination_1.",
    )
    parser.add_argument(
        "--class_weight",
        "-cw",
        default="False",
        choices=["True", "False"],
        help="Wether to compute class weights and use them in loss function. Beware, only works for models with single output.",
    )
    parser.add_argument(
        "--target",
        "-t",
        choices=["Illumination", "Rain", "Snow", "Fog", "Timing", "All"],
        default="Illumination",
        help="Label to use as target (softmax will be used in final layer). If 'All', will use all columns and sigmoid activation in final layer.",
    )
    parser.add_argument(
        "--random_state",
        "-rs",
        type=int,
        default=22,
        help="Random state to use for splitting data into train and validation sets.",
    )
    parser.add_argument(
        "--warmup",
        "-wa",
        type=int,
        default=0,
        help="Number of epochs to use as warm-up stage, (a higher initial learning rate will be used)",
    )
    parser.add_argument(
        "--init_lr",
        "-lr",
        type=float,
        default=0.003,
        help="Value of initial learning rate. Note: an exponential decay will be applied to reach final learning rate of 1e-5 at the last epoch",
    )
    parser.add_argument(
        "--dropout",
        "-dr",
        type=float,
        default=0.5,
        help="Dropout rate to use before final layer",
    )

    return parser
