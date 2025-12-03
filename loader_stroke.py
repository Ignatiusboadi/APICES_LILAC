from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import torchio as tio
import numpy as np
import random
from itertools import permutations
from typing import List, Tuple
from collections import defaultdict
import random


def create_index_pairs(df, subject_id="subject", time_point="ScanOrder"):
    subject_list = []
    time_point_count_list = []
    for group_id, group_df in df.groupby(subject_id):
        subject_list.append(group_id)
        time_point_count_list.append(len(np.unique(group_df[time_point].to_list())))
    return np.array(subject_list), np.array(time_point_count_list)


def create_batches(
        class_names: List[str],
        instance_counts: List[int],
        batch_size: int,
        m: int,
) -> List[List[Tuple[str, int]]]:
    """
    Creates batches while minimizing class splits.
    
    Each batch is a list of tuples (class_name, number_of_instances).
    """
    from collections import deque

    # Create a queue of (class_name, remaining_count)
    class_list = list(zip(class_names, instance_counts))
    random.shuffle(class_list)
    queue = deque(class_list)
    batches = []

    while queue:
        current_batch = []
        remaining_space = batch_size
        temp_queue = deque()

        while queue and remaining_space > 0:
            cls, count = queue.popleft()

            if count <= remaining_space:
                # Take the whole class if it fits
                current_batch.append((cls, count))
                remaining_space -= count
            elif count > m or remaining_space < 2:
                # If allowed to split OR not enough space to take 2
                # Take as much as possible but leave at least 2 behind
                if remaining_space >= 2 and count - remaining_space >= 2:
                    current_batch.append((cls, remaining_space))
                    temp_queue.appendleft((cls, count - remaining_space))
                    remaining_space = 0
                else:
                    temp_queue.append((cls, count))
            else:
                # If we cannot split and it's not over m, defer this class
                temp_queue.append((cls, count))

        # Push back the remaining classes
        queue = temp_queue + queue
        batches.append(current_batch)
    #print(len(queue))
    #print(len(temp_queue))
    return batches


def top_off_batches(
        class_names: List[str],
        instance_counts: List[int],
        batch_size: int,
        batches: List[List[Tuple[str, int]]],
        verbose: bool = False
) -> List[List[Tuple[str, int]]]:
    """
    Goes through batches and tops them off using classes already present in the batch,
    if those classes still have remaining instances not yet used.
    Prints messages when batches are filled or not fillable.
    """
    from collections import defaultdict

    # Track how many instances of each class have been used across all batches
    used_instances = defaultdict(int)

    # Map class name to total available
    class_total = dict(zip(class_names, instance_counts))

    new_batches = []

    for batch_idx, batch in enumerate(batches):
        batch_total = sum(count for _, count in batch)
        remaining_space = batch_size - batch_total

        if remaining_space <= 0:
            new_batches.append(batch)
            for cls, count in batch:
                used_instances[cls] += count
            continue  # Batch is already full

        # Check which classes in this batch have unused instances
        new_batch = batch.copy()
        expanded = False

        # Sort to prioritize classes with the most leftover instances
        for cls, current_count in sorted(batch, key=lambda x: class_total[x[0]] - used_instances[x[0]], reverse=True):
            available = class_total[cls] - used_instances[cls]
            if available <= 0:
                continue

            # Max we can take without going over batch size or available
            take = min(available, remaining_space)

            # Ensure we never add only 1 instance
            if take == 1:
                continue
            if take >= 2:
                # Update batch and tracking
                for i, (c, cnt) in enumerate(new_batch):
                    if c == cls:
                        new_batch[i] = (c, cnt + take)
                        break
                used_instances[cls] += take
                remaining_space -= take
                expanded = True
                if verbose:
                    print(
                        f"Topped off batch {batch_idx + 1} with {take} instance(s) of '{cls}' (max available: {class_total[cls]})")

            if remaining_space <= 0:
                break

        if not expanded:
            if verbose:
                print(
                    f"Could not top off batch {batch_idx + 1} (missing {remaining_space} instances). No available extra instances in present classes.")

            # Still need to update used_instances with current batch
        for cls, count in new_batch:
            used_instances[cls] += count

        new_batches.append(new_batch)

    return new_batches


def swap_whole_classes_to_fill(
        class_names: List[str],
        instance_counts: List[int],
        batch_size: int,
        batches: List[List[Tuple[str, int]]],
        verbose: bool = False
) -> List[List[Tuple[str, int]]]:
    """
    Swaps whole classes between underfilled batches to help complete at least one batch.
    No class is split. If a batch becomes more incomplete, it is filled with a new class if possible.
    """
    from collections import defaultdict

    # Map class name to total available
    class_total = dict(zip(class_names, instance_counts))

    # Track how many instances of each class have been used across all batches
    used_instances = defaultdict(int)
    for batch in batches:
        for cls, count in batch:
            used_instances[cls] += count

    # Identify underfilled batches
    incomplete_batches = []
    for idx, batch in enumerate(batches):
        total = sum(c for _, c in batch)
        if total < batch_size:
            incomplete_batches.append((idx, batch, total))

    modified_batches = [list(batch) for batch in batches]  # Make deep copies
    success = False
    for i in range(len(incomplete_batches)):
        idx_a, batch_a, total_a = incomplete_batches[i]
        for j in range(i + 1, len(incomplete_batches)):
            idx_b, batch_b, total_b = incomplete_batches[j]

            # Try all class pairs (whole-class swaps)
            for cls_a, count_a in batch_a:
                for cls_b, count_b in batch_b:
                    new_total_a = total_a - count_a + count_b
                    new_total_b = total_b - count_b + count_a

                    if new_total_a == batch_size:
                        # Perform swap
                        new_batch_a = [(c, cnt) for c, cnt in batch_a if c != cls_a]
                        new_batch_b = [(c, cnt) for c, cnt in batch_b if c != cls_b]
                        new_batch_a.append((cls_b, count_b))
                        new_batch_b.append((cls_a, count_a))
                        if verbose:
                            print(
                                f"Swapped class '{cls_a}' from batch {idx_a + 1} with class '{cls_b}' from batch {idx_b + 1} to fill batch {idx_a + 1}",
                                flush=True)

                        # Update used_instances doesn't change — same total use

                        # Attempt to fill now-more-incomplete batch B
                        total_b_new = new_total_b
                        used_classes_b = {cls for cls, _ in new_batch_b}

                        for cls in class_names:
                            available = class_total[cls] - used_instances[cls]
                            if cls not in used_classes_b and available >= 2:
                                fill = min(available, batch_size - total_b_new)
                                if fill >= 2:
                                    new_batch_b.append((cls, fill))
                                    used_instances[cls] += fill
                                    if verbose:
                                        print(f"Filled batch {idx_b + 1} with {fill} of new class '{cls}'")
                                break

                        # Save modified batches
                        modified_batches[idx_a] = new_batch_a
                        modified_batches[idx_b] = new_batch_b
                        success = True
                        return modified_batches, success  # Exit after first successful swap
    if verbose:
        print("No valid whole-class swaps found to complete any batch.")
    return modified_batches, success


def randomly_fill_remaining(
        class_names: List[str],
        instance_counts: List[int],
        batch_size: int,
        batches: List[List[Tuple[str, int]]],
        verbose: bool = False
) -> List[List[Tuple[str, int]]]:
    """
    Fills any underfilled batches with a random NEW class (not yet in the batch)
    that has enough available instances. Does not allow singleton additions.
    """
    # Map class to total available
    class_total = dict(zip(class_names, instance_counts))

    # Track how many instances of each class are already used
    used_instances = defaultdict(int)
    for batch in batches:
        for cls, count in batch:
            used_instances[cls] += count

    # Deep copy to modify
    modified_batches = [list(batch) for batch in batches]

    for batch_idx, batch in enumerate(modified_batches):
        current_total = sum(count for _, count in batch)
        remaining_space = batch_size - current_total

        if remaining_space <= 0:
            continue  # Already full

        used_in_batch = {cls for cls, _ in batch}

        # Shuffle candidates to add randomness
        candidates = list(set(class_names) - used_in_batch)
        random.shuffle(candidates)

        for new_cls in candidates:
            available = class_total[new_cls] - used_instances[new_cls]
            if available >= 2:
                to_add = min(available, remaining_space)
                if to_add >= 2:
                    batch.append((new_cls, to_add))
                    used_instances[new_cls] += to_add
                    if verbose:
                        print(f"Randomly filled batch {batch_idx + 1} with {to_add} of NEW class '{new_cls}'")
                    break  # Only fill once per batch

    return modified_batches


def fill_remaining_with_duds(
        class_names: List[str],
        instance_counts: List[int],
        batch_size: int,
        batches: List[List[Tuple[str, int]]]
) -> List[List[Tuple[str, int]]]:
    """
    Fills any underfilled batches with a filler class that gets removed during evaluation
    """
    # Map class to total available
    class_total = dict(zip(class_names, instance_counts))

    # Track how many instances of each class are already used
    used_instances = defaultdict(int)
    for batch in batches:
        for cls, count in batch:
            used_instances[cls] += count

    # Deep copy to modify
    modified_batches = [list(batch) for batch in batches]

    for batch_idx, batch in enumerate(modified_batches):
        current_total = sum(count for _, count in batch)
        remaining_space = batch_size - current_total

        if remaining_space <= 0:
            continue  # Already full
        else:
            batch.append(("dud", remaining_space))

    return modified_batches


def validate_batches(
        class_names: List[str],
        instance_counts: List[int],
        batches: List[List[Tuple[str, int]]]
):
    """
    Validates that all data from initial class list and counts is properly used
    in the batches, and gives a breakdown of usage and splits.
    """
    print("\nRunning batch validation...\n")

    # Map class to total available
    total_available = dict(zip(class_names, instance_counts))

    # Track total used per class
    used_counts = defaultdict(int)

    # Track in which batches each class appears
    class_to_batches = defaultdict(list)

    for batch_idx, batch in enumerate(batches):
        for cls, count in batch:
            used_counts[cls] += count
            class_to_batches[cls].append((batch_idx + 1, count))

    all_classes = set(class_names) | set(used_counts.keys())

    missing_data = False

    print("Class Usage Summary:")
    for cls in sorted(all_classes):
        used = used_counts.get(cls, 0)
        available = total_available.get(cls, 0)
        status = "OK" if used == available else ("MISSING" if used < available else "OVERUSED")
        print(f"  - {cls:15} Used: {used:3} / Available: {available:3} → {status}")
        if used != available:
            missing_data = True

    if not missing_data:
        print("\nAll instances are correctly assigned — no loss or overuse.")
    else:
        print("\nInstance counts mismatch detected — check above for details.")

    # Check for class splits
    split_classes = {cls: locs for cls, locs in class_to_batches.items() if len(locs) > 1}

    print(f"\nSplit Class Summary:")
    print(f"  - Total classes split across batches: {len(split_classes)}")
    if split_classes:
        for cls, locations in split_classes.items():
            loc_str = ", ".join([f"batch {idx} ({count})" for idx, count in locations])
            print(f"    - {cls}: {loc_str}")
    else:
        print("  - No classes were split across batches.")
    print("\nBatch validation complete.\n")


def sample_ids_from_batches(
        df: pd.DataFrame,
        batches: List[List[Tuple[str, int]]],
        image_id="fname",
        subject_column="subject",
        time_point_column="ScanOrder"
) -> List[List[int]]:
    """
    Given a DataFrame with 'subject', 'timepoint', and 'id' columns, and a list of batches
    containing (subject, count) pairs, sample appropriate rows (by index) for each batch.
    """
    # Track used timepoints per subject across batches
    used_timepoints = {subject: set() for subject in df['subject'].unique()}
    #classes, counts = np.unique(df['image_id'], return_counts=True)
    #print(classes[counts>1])
    #print(counts[counts>1])

    # Ensure uniqueness
    #assert df['image_id'].is_unique, "Column 'id' must be unique per row"
    # Create a fast lookup for subject -> timepoint -> list of ids
    #print(df)
    subj_tp_to_ids = (
        df.groupby([subject_column, time_point_column])[image_id]
        .apply(list)
        .to_dict()
    )

    # Create lookup from id to index for returning row indices
    id_to_index = df.set_index(image_id).index.to_series().to_dict()
    id_to_row_index = df.reset_index().set_index(image_id)['index'].to_dict()

    final_batches = []

    for batch_idx, batch in enumerate(batches):
        batch_indices = []

        for subject, count in batch:
            # in the benchmarking scenario, we want to fill up the batches with duds
            if subject == "dud":
                for i in range(count):
                    batch_indices.append("dud")
            else:
                # All timepoints available for this subject
                all_timepoints = set(tp for (subj, tp) in subj_tp_to_ids if subj == subject)

                # Timepoints not yet used globally
                unused_global = list(all_timepoints - used_timepoints[subject])
                random.shuffle(unused_global)

                selected_timepoints = []

                # Prioritize unseen timepoints
                take = min(len(unused_global), count)
                selected_timepoints.extend(unused_global[:take])

                # If not enough, pick from remaining (but not already in current batch for this subject)
                remaining_needed = count - len(selected_timepoints)
                if remaining_needed > 0:
                    already_used_in_batch = set(selected_timepoints)
                    remaining_timepoints = list(all_timepoints - already_used_in_batch)
                    if remaining_timepoints:
                        random.shuffle(remaining_timepoints)
                        to_add = remaining_timepoints[:remaining_needed]
                        selected_timepoints.extend(to_add)

                # Register used timepoints globally
                used_timepoints[subject].update(selected_timepoints)

                # For each timepoint, randomly choose one ID and get the row index
                for tp in selected_timepoints:
                    possible_ids = subj_tp_to_ids[(subject, tp)]
                    selected_id = random.choice(possible_ids)
                    row_index = id_to_row_index[selected_id]
                    batch_indices.append(row_index)

        final_batches.append(batch_indices)

    return final_batches


def sample_subjects_batch_specific(meta_data_df, batchsize, max_class_size, subject_id="subject",
                                   time_point="ScanOrder", verbose=False, testing=False):
    subject_list, count_list = create_index_pairs(meta_data_df, subject_id, time_point)
    # sort out those subjects with only one image
    subject_list = list(subject_list[count_list > 1])
    count_list = list(count_list[count_list > 1])

    batches = create_batches(subject_list, count_list, batchsize, max_class_size)
    if not testing:
        batches = top_off_batches(subject_list, count_list, batchsize, batches, verbose)
        success = True
        while success:
            batches, success = swap_whole_classes_to_fill(subject_list, count_list, batchsize, batches, verbose)
        final_batches = randomly_fill_remaining(subject_list, count_list, batchsize, batches, verbose)

    else:
        final_batches = fill_remaining_with_duds(subject_list, count_list, batchsize, batches)

    if verbose:
        validate_batches(subject_list, count_list, final_batches)
    row_id_batches = sample_ids_from_batches(meta_data_df, final_batches)

    # split up the batches to a simple oine dimensional list
    # be careful to shuffle it anymore
    return [x for batch in row_id_batches for x in batch]


class single_image_multiple_target_loader3D_stroke(Dataset):
    def __init__(self, args, trainvaltest, file_column='fname', subject_id="subject", time_point="ScanOrder",
                 test_df=None, extra_meta=None):
        self.attribute_list = args.attribute_list
        self.trainvaltest = trainvaltest
        if trainvaltest == 'train':
            self.meta_data_df = pd.read_csv(args.csv_file_train).iloc[:39]
            self.augmentation = True
            assert set([file_column, subject_id, time_point] + self.attribute_list + extra_meta).issubset(
                set(list(self.meta_data_df.columns))), f"Check input csv file column names"
        elif trainvaltest == 'val':
            self.meta_data_df = pd.read_csv(args.csv_file_val).iloc[:30]
            self.augmentation = False
            assert set([file_column, subject_id, time_point] + self.attribute_list + extra_meta).issubset(
                set(list(self.meta_data_df.columns))), f"Check input csv file column names"
        else:
            if test_df is not None:
                self.meta_data_df = test_df
            else:
                self.meta_data_df = pd.read_csv(args.csv_file_test)  # .iloc[:9]

            self.augmentation = False
            assert set([file_column, subject_id, time_point] + self.attribute_list).issubset(
                set(list(self.meta_data_df.columns))), f"Check input csv file column names"
        self.batchsize = args.batchsize
        self.max_class_size = args.max_class_size
        self.image_size = args.image_size
        self.imgdir = args.image_directory
        self.subject_id = subject_id
        self.file_column = file_column
        self.time_point = time_point
        self.rescale_intensity = args.rescale_intensity
        if extra_meta is None:
            extra_meta = []
        self.extra_meta = extra_meta
        self.meta_data_df = self.meta_data_df[
            np.unique([file_column, subject_id, time_point] + self.attribute_list + self.extra_meta).tolist()]
        # print("size df before second dropping NaNs: {}".format(len(self.meta_data_df)))
        self.meta_data_df = self.meta_data_df.dropna(axis=0).reset_index(drop=True)
        # print("size df after second dropping NaNs: {}".format(len(self.meta_data_df)), flush=True)
        self.index_combination, self.attribute_dict = create_index_pairs(self.meta_data_df, self.subject_id,
                                                                         self.time_point)
        self.index_list = sample_subjects_batch_specific(self.meta_data_df, self.batchsize, self.max_class_size,
                                                         self.subject_id, self.time_point,
                                                         testing=not self.augmentation)

        # TODO new stuff like group wise transformations
        self.group_transformation_dict = {}
        self.set_up_groupwise_transformations()

    # given that images are not loaded in pairs, overarching augmentations are done on a per subject level
    # when doing multiple iterations, repeatedly calling this function will create new augmentations
    def set_up_groupwise_transformations(self):
        for specific_subject_id in self.meta_data_df[self.subject_id].to_list():
            self.group_transformation_dict[specific_subject_id] = {}
            self.group_transformation_dict[specific_subject_id]["use_groupwise"] = np.random.randint(0, 2)
            self.group_transformation_dict[specific_subject_id]["use_affine"] = np.random.randint(0, 2)
            self.group_transformation_dict[specific_subject_id]["affine_degree"] = tuple(
                np.random.uniform(low=-40, high=40, size=3))
            self.group_transformation_dict[specific_subject_id]["affine_translate"] = tuple(
                np.random.uniform(low=-10, high=10, size=3))
            self.group_transformation_dict[specific_subject_id]["use_flip"] = np.random.randint(0, 2)

    def reshuffle_dataset(self, verbose=False):
        self.index_combination, self.attribute_dict = create_index_pairs(self.meta_data_df, self.subject_id,
                                                                         self.time_point)
        self.index_list = sample_subjects_batch_specific(self.meta_data_df, self.batchsize, self.max_class_size,
                                                         self.subject_id, self.time_point, verbose)
        self.set_up_groupwise_transformations()

    def __getitem__(self, index):
        row_index = self.index_list[index]
        if row_index == "dud":
            image = np.zeros((1, *self.image_size)).astype('float')
            target_list = []
            for target in self.attribute_list:
                target_value = 0
                target_list.append(target_value)
            specific_subject_id = "dud"
            # print('dud entry', self.extra_meta)
            return image, target_list, {
                item: (specific_subject_id if item == 'subject' else 0) for item in ['subject'] + self.extra_meta}

        df_row = self.meta_data_df.iloc[[row_index]]

        target_list = []
        for target in self.attribute_list:
            target_value = df_row[target].item()
            target_list.append(target_value)

        fname = os.path.join(self.imgdir, df_row[self.file_column].item())
        specific_subject_id = df_row[self.subject_id].item()

        image = tio.ScalarImage(fname)
        resize = tio.transforms.Resize(tuple(self.image_size))
        image = resize(image)
        pairwise_transform_list = []
        imagewise_transform_list = []

        if self.augmentation:

            if self.group_transformation_dict[specific_subject_id]["use_groupwise"]:
                if self.group_transformation_dict[specific_subject_id]["use_affine"]:
                    affine_degree = self.group_transformation_dict[specific_subject_id]["affine_degree"]
                    affine_translate = self.group_transformation_dict[specific_subject_id]["affine_translate"]
                    pairwise_transform_list.append(tio.Affine(scales=(1, 1, 1),
                                                              degrees=affine_degree,
                                                              translation=affine_translate,
                                                              image_interpolation='linear',
                                                              default_pad_value='minimum'))

                if self.group_transformation_dict[specific_subject_id]["use_flip"]:
                    pairwise_transform_list.append(tio.Flip(axes=('LR',)))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomNoise(mean=0, std=2))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomGamma(0.3))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomBlur(2))

        # TODO warning, flipped group and image wise order, also added intensity scaling
        if self.rescale_intensity:
            imagewise_transform_list.append(tio.RescaleIntensity(out_min_max=(0, 1)))

        if len(imagewise_transform_list) > 0:
            imagewise_augmentation = tio.Compose(imagewise_transform_list)
            image = imagewise_augmentation(image)
        if len(pairwise_transform_list) > 0:
            pairwise_augmentation = tio.Compose(pairwise_transform_list)
            image = pairwise_augmentation(image)

        image = image.numpy().astype('float')
        meta_dict = {key: df_row[key].item() for key in [self.subject_id] + self.extra_meta}
        # if self.trainvaltest == 'val':
            # print('val meta_dict keys', meta_dict.keys())

        return image, target_list, meta_dict

    def __len__(self):
        return len(self.index_list)


