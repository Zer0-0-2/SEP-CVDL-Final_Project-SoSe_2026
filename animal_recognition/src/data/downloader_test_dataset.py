"""
Creates a test dataset of exactly 100 images for 20 animal breeds.
The images are pulled from vetted huggingface datasets (Oxford Pets, Stanford Dogs and Imagenet1k)
and saved in the same folder structure as the trainin dataset.

NOTE/WARNING TO ANYONE RUNNING THIS: this script is incredibly unoptimized because it has to stream through imagenet1k instead of being
able to download the images directly by label.
It is incredibly memory intensive (took my 32 gb of ram and 20gb of swap memory) and took 30 minutes to run.
Since this is a one time operation I did not bother optimizing it, since i will be sharing the zipped dataset with the team anyway.
"""

import os
from PIL import Image
from datasets import load_dataset

DESTINATION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_dataset")
)
TARGET_PER_CLASS = 100

CLASSES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Maine_Coon",
    "Ragdoll",
    "Sphynx",
    "Tabby",
    "Tiger_Cat",
    "Beagle",
    "Pug",
    "Boxer",
    "Shiba_Inu",
    "Samoyed",
    "Golden_Retriever",
    "German_Shepherd",
    "Siberian_Husky",
    "Dalmatian",
    "Rottweiler",
]


def ensure_rgb(image_to_save):
    if image_to_save.mode != "RGB":
        return image_to_save.convert("RGB")
    return image_to_save


def pull_from_huggingface(
    huggingface_repo,
    dataset_split,
    index_to_label_mapping,
    label_column_name,
    image_column_name,
    images_downloaded_per_class,
    save_image_function,
    is_streaming=False,
    resolve_integer_labels_to_strings=True,
    images_to_skip_per_class=None,
    target_image_counts_per_class=None,
):
    # for tiger cat skip first 200 images because they are already in the training dataset
    if images_to_skip_per_class is None:
        images_to_skip_per_class = {}
    if target_image_counts_per_class is None:
        target_image_counts_per_class = {}

    print(f"Loading {huggingface_repo}...")

    # Create reverse lookup to find the target class index from the dataset's native label
    label_to_class_index = {
        label: class_index for class_index, label in index_to_label_mapping.items()
    }
    dataset = load_dataset(huggingface_repo, split=dataset_split, streaming=is_streaming)

    for image_record in dataset:
        current_image_label = image_record[label_column_name]

        if resolve_integer_labels_to_strings:
            current_image_label = dataset.features[label_column_name].int2str(current_image_label)

        if current_image_label in label_to_class_index:
            class_index = label_to_class_index[current_image_label]
            current_target_count = target_image_counts_per_class.get(class_index, TARGET_PER_CLASS)

            if images_to_skip_per_class.get(class_index, 0) > 0:
                images_to_skip_per_class[class_index] -= 1
                continue

            if images_downloaded_per_class[class_index] < current_target_count:
                save_image_function(image_record[image_column_name], class_index)
                images_downloaded_per_class[class_index] += 1

                if (
                    is_streaming
                    and sum(images_downloaded_per_class[k] for k in index_to_label_mapping.keys())
                    % 10
                    == 0
                ):
                    print(
                        f"Streaming progress for {huggingface_repo}: {images_downloaded_per_class[class_index]}/{current_target_count} for class {class_index}"
                    )

        # Early exit if we have found enough images for all requested classes
        if all(
            images_downloaded_per_class[k] >= target_image_counts_per_class.get(k, TARGET_PER_CLASS)
            for k in index_to_label_mapping.keys()
        ):
            break


def main():
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    images_downloaded_per_class = {class_index: 0 for class_index in range(20)}
    current_image_number = 1

    def save_image(image_to_save, class_index):
        nonlocal current_image_number
        image_to_save = ensure_rgb(image_to_save)
        class_name = CLASSES[class_index]

        # Create folder structure for this class
        class_directory = os.path.join(DESTINATION_DIR, class_name)
        os.makedirs(class_directory, exist_ok=True)

        image_filename = f"{current_image_number:04d}.jpg"
        image_filepath = os.path.join(class_directory, image_filename)
        image_to_save.save(image_filepath, format="JPEG")

        current_image_number += 1

    # Oxford Pets
    pull_from_huggingface(
        huggingface_repo="cvdl/oxford-pets",
        dataset_split="train+test+valid",
        index_to_label_mapping={
            0: "Abyssinian",
            1: "Bengal",
            2: "Birman",
            3: "Bombay",
            4: "British Shorthair",
            5: "Maine Coon",
            6: "Ragdoll",
            7: "Sphynx",
            10: "beagle",
            11: "pug",
            12: "boxer",
            13: "shiba inu",
            14: "samoyed",
        },
        label_column_name="category",
        image_column_name="img",
        images_downloaded_per_class=images_downloaded_per_class,
        save_image_function=save_image,
    )

    # Stanford Dogs
    pull_from_huggingface(
        huggingface_repo="maurice-fp/stanford-dogs",
        dataset_split="train+test",
        index_to_label_mapping={
            15: "n02099601-golden_retriever",
            16: "n02106662-German_shepherd",
            17: "n02110185-Siberian_husky",
            19: "n02106550-Rottweiler",
        },
        label_column_name="label",
        image_column_name="image",
        images_downloaded_per_class=images_downloaded_per_class,
        save_image_function=save_image,
    )

    # Tabby, Dalmatian and Tiger Cat

    # note that the first 200 tiger_cat images are skipped and then 250 are downloaded, because for our training data we
    # already pulle the first 200 tiger_cat imgages from imagenet_1k
    pull_from_huggingface(
        huggingface_repo="ILSVRC/imagenet-1k",
        dataset_split="train",
        index_to_label_mapping={8: 281, 18: 251, 9: 282},
        label_column_name="label",
        image_column_name="image",
        images_downloaded_per_class=images_downloaded_per_class,
        save_image_function=save_image,
        is_streaming=True,
        resolve_integer_labels_to_strings=False,
        images_to_skip_per_class={9: 200},
        target_image_counts_per_class={9: 250},
    )

    print(f"\nDone! Dataset created in {DESTINATION_DIR}")
    print("Class counts:")
    for class_index, class_name in enumerate(CLASSES):
        print(f"  {class_name} ({class_index}): {images_downloaded_per_class.get(class_index, 0)}")


if __name__ == "__main__":
    main()
