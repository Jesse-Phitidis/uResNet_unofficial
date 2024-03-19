import torchio as tio
import torch
import numpy as np


# It might actually be better to precompute the probability maps and instead inherit from WeightedSampler
class LabelSampler_translation(tio.data.LabelSampler):

    '''

    This class extends torchio.data.LabelSampler by randomly translating the patch to be extracted as per the original paper.

    :arg fraction: The fraction of the patch size up to which we can translate. Default 0.5. Do not increase. 

    Note: The label image must be named 'label'

    '''

    def __init__(
            self,
            patch_size,
            label_name = None,
            label_probabilities = None,
            fraction = (0.5, 0.5, 0.5),
            padding_mode_image = 'edge',
            padding_mode_label = 0,
            one_hot = False,
    ):
        super().__init__(patch_size, label_name, label_probabilities)
        self.num_classes = len(label_probabilities)
        self.frac = fraction
        self.padding_mode_image = padding_mode_image
        self.padding_mode_label = padding_mode_label
        self.one_hot = one_hot

        if np.any(np.array(fraction) > 0.5):
            raise ValueError('Translation by more than half the patch size may result in the class no longer being within the patch')

    def extract_patch(
            self,
            subject,
            probability_map,
            cdf,
    ):
        i, j, k = self.get_random_index_ini(probability_map, cdf)
        index_ini = i, j, k
        si, sj, sk = self.patch_size.astype(np.int32) # I added this
        patch_size = si, sj, sk

        # Randomly translate index_ini by up to fraction of the patch size so that classes are not always centred
        small_vals = [1e-6 if i != 0 else 0 for i in self.frac]
        si_lb, sj_lb, sk_lb = -int((np.floor(si*self.frac[0] - small_vals[0]))), -int((np.floor(sj*self.frac[1] - small_vals[1]))), -int((np.floor(sk*self.frac[2] - small_vals[2])))
        si_ub, sj_ub, sk_ub = int((np.floor(si*self.frac[0] - small_vals[0]))+1), int((np.floor(sj*self.frac[1] - small_vals[1]))+1), int((np.floor(sk*self.frac[2] - small_vals[2]))+1)
        trans_i, trans_j, trans_k = torch.randint(si_lb, si_ub, (1,)).item(), torch.randint(sj_lb, sj_ub, (1,)).item(), torch.randint(sk_lb, sk_ub, (1,)).item()
        rand_trans = np.array((trans_i, trans_j, trans_k))
        index_ini += rand_trans
        index_fin = index_ini + patch_size

        # If this results in the patch being out of bounds then zero pad the subject accordingly
        shape = np.array(subject.spatial_shape)
        lower_padding = np.array([np.abs(ini) if ini < 0 else 0 for ini in index_ini])
        upper_padding = np.array([fin - dim if fin > dim else 0 for fin, dim in zip(index_fin, shape)])
        index_ini_new = index_ini + lower_padding
        
        padding = []
        for ini, fin in zip(lower_padding, upper_padding):
            padding.append(ini)
            padding.append(fin) 
        padding = tuple(padding)

        padding_transform_image = tio.transforms.Pad(padding, padding_mode = self.padding_mode_image)
        padding_transform_label = tio.transforms.Pad(padding, padding_mode = self.padding_mode_label)
        
        subject_label = subject['label']
        subject_label_padded = padding_transform_label(subject_label)
        subject = padding_transform_image(subject)
        subject.add_image(subject_label_padded, 'label')

        # Convert to OneHot (not done in composed transforms because zero padding OneHot tensor results in classless voxels which is an issue for the standard loss functions)
        if self.one_hot:
            one_hot_transform = tio.transforms.OneHot(num_classes=self.num_classes)
            subject = one_hot_transform(subject)

        cropped_subject = self.crop(
            subject,
            index_ini_new,
            patch_size,
        )

        # Location of the crop in reference to the original image (x_ini, x_fin, y_ini, y_fin, z_ini, z_fin)
        cropped_subject.location = torch.from_numpy(np.append(index_ini, index_fin, axis = 0)).type(torch.int64)
        return cropped_subject 



# This makes the single improvement of not clearing prob boarders so we do not need to pad the image with half the patch size prior to training
class LabelSampler_translation_no_boarders(LabelSampler_translation):

    def __init__(
            self,
            patch_size,
            label_name = None,
            label_probabilities = None,
            fraction = (0.5, 0.5, 0.5),
            padding_mode_image = 'edge',
            padding_mode_label = 0,
            one_hot = False
    ):

        super().__init__(patch_size, label_name, label_probabilities, fraction, padding_mode_image, padding_mode_label, one_hot)

    def __call__(
            self,
            subject,
            num_patches = None,
    ):
    
        ''' Commented out because will deal with this in my extract_patch method'''

        subject.check_consistent_space()
        # if np.any(self.patch_size > subject.spatial_shape):
        #     message = (
        #         f'Patch size {tuple(self.patch_size)} cannot be'
        #         f' larger than image size {tuple(subject.spatial_shape)}'
        #     )
        #     raise RuntimeError(message)
        kwargs = {} if num_patches is None else {'num_patches': num_patches}
        return self._generate_patches(subject, **kwargs)

    def get_random_index_ini(
            self,
            probability_map,
            cdf,
    ):

        ''' Commented out assertions since we deal with this in my extract_patch method '''

        center = self.sample_probability_map(probability_map, cdf)
        #assert np.all(center >= 0)
        # See self.clear_probability_borders
        index_ini = center - self.patch_size // 2
        #assert np.all(index_ini >= 0)
        return index_ini
    
    @staticmethod
    def clear_probability_borders_new(
            probability_map,
            patch_size,
    ):
        # We don't want to clear the boarders
        pass

    @staticmethod
    def get_probabilities_from_label_map_new(
            label_map,
            label_probabilities_dict,
            patch_size,
    ):
        """Create probability map according to label map probabilities."""
        # This also used to clear the boarders in LabelSampler so we remove that bit

        multichannel = label_map.shape[0] > 1
        probability_map = torch.zeros_like(label_map)
        label_probs = torch.Tensor(list(label_probabilities_dict.values()))
        normalized_probs = label_probs / label_probs.sum()
        iterable = zip(label_probabilities_dict, normalized_probs)
        for label, label_probability in iterable:
            if multichannel:
                mask = label_map[label]
            else:
                mask = label_map == label
            label_size = mask.sum()
            if not label_size:
                continue
            prob_voxels = label_probability / label_size
            if multichannel:
                probability_map[label] = prob_voxels * mask
            else:
                probability_map[mask] = prob_voxels
        if multichannel:
            probability_map = probability_map.sum(dim=0, keepdim=True)
        return probability_map


    # Methods below are unchanged except for calling my static methods instead of the inherited ones

    def get_probability_map(self, subject):
            label_map_tensor = self.get_probability_map_image(subject).data.float()

            if self.label_probabilities_dict is None:
                return label_map_tensor > 0
            probability_map = self.get_probabilities_from_label_map_new(
                label_map_tensor,
                self.label_probabilities_dict,
                self.patch_size,
            )
            return probability_map

    def process_probability_map(
                self,
                probability_map,
                subject,
        ):
            # Using float32 can create cdf with maximum very far from 1, e.g. 0.92!
            data = probability_map[0].numpy().astype(np.float64)
            assert data.ndim == 3
            self.clear_probability_borders_new(data, self.patch_size)
            total = data.sum()
            if total == 0:
                half_patch_size = tuple(n // 2 for n in self.patch_size)
                message = (
                    'Empty probability map found:'
                    f' {self.get_probability_map_image(subject).path}'
                    '\nVoxels with positive probability might be near the image'
                    ' border.\nIf you suspect that this is the case, try adding a'
                    ' padding transform\nwith half the patch size:'
                    f' torchio.Pad({half_patch_size})'
                )
                raise RuntimeError(message)
            data /= total  # normalize probabilities
            return data