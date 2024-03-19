from os.path import join
from os import listdir
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchio as tio

from data.samplers.patch_trans_sampler import LabelSampler_translation_no_boarders as Sampler

hold_out_val = [
    'MSS_591_0000.nii.gz', 'MSS_656_0000.nii.gz', 'MSS_705_0000.nii.gz', 'MSS_665_0000.nii.gz', 'MSS_690_0000.nii.gz', 'MSS_714_0000.nii.gz', 
    'MSS_584_0000.nii.gz', 'MSS_722_0000.nii.gz', 'MSS_517_0000.nii.gz', 'MSS_617_0000.nii.gz', 'MSS_730_0000.nii.gz', 'MSS_516_0000.nii.gz', 
    'MSS_620_0000.nii.gz', 'MSS_663_0000.nii.gz', 'MSS_676_0000.nii.gz', 'MSS_622_0000.nii.gz', 'MSS_620_0000.nii.gz', 'MSS_682_0000.nii.gz', 
    'MSS_659_0000.nii.gz', 'MSS_561_0000.nii.gz', 'MSS_599_0000.nii.gz', 'MSS_594_0000.nii.gz', 'MSS_610_0000.nii.gz', 'MSS_573_0000.nii.gz', 
    'MSS_657_0000.nii.gz', 'MSS_573_0000.nii.gz', 'MSS_561_0000.nii.gz', 'MSS_542_0000.nii.gz', 'MSS_689_0000.nii.gz', 'MSS_586_0000.nii.gz'
    ]

def get_label_name_from_image(image_name):
    return image_name[:7] + '.nii.gz'

class uResNet_DataModule(pl.LightningDataModule):
    def __init__(self, data_root, patch_size, transforms_train, val_run):
        super().__init__()
        self.data_root = data_root
        self.patch_size = patch_size
        self.transforms_train = transforms_train
        self.val_run = val_run

        self.save_hyperparameters()

    def build_dataset(self, stage):
        
        assert stage in ['train', 'test', 'val']
        postfix = 'Tr' if stage == 'train' else 'Ts'
        if stage == 'train' and self.val_run == True:
            return self.build_train_for_val_run()
        if stage == 'val' and self.val_run == True:
            return self.build_val()

        images_folder = join(self.data_root, 'images' + postfix)
        images = join(images_folder, '%s')
        labels = join(self.data_root, 'labels' + postfix, '%s')

        subjects_list = []
        for image_name in sorted(listdir(images_folder)):
            label_name = get_label_name_from_image(image_name)
            num = image_name[4:7]
            subject = tio.Subject(
                image=tio.ScalarImage(images % image_name),
                label=tio.LabelMap(labels % label_name),
                id=num 
            )
            subjects_list.append(subject)
        
        return tio.SubjectsDataset(subjects_list, self.transforms_train if stage == 'train' else None)

    def build_train_for_val_run(self):
        postfix = 'Tr'
        images_folder = join(self.data_root, 'images' + postfix)
        images = join(images_folder, '%s')
        labels = join(self.data_root, 'labels' + postfix, '%s')

        subjects_list = []
        for image_name in sorted(listdir(images_folder)):
            if image_name not in hold_out_val:
                label_name = get_label_name_from_image(image_name)
                num = image_name[4:7]
                subject = tio.Subject(
                    image = tio.ScalarImage(images % image_name),
                    label = tio.LabelMap(labels % label_name),
                    id = num 
                )
                subjects_list.append(subject)
        
        return tio.SubjectsDataset(subjects_list, self.transforms_train)

    def build_val(self):

        if self.val_run is False:
            return

        postfix = 'Tr'
        images_folder = join(self.data_root, 'images' + postfix)
        images = join(images_folder, '%s')
        labels = join(self.data_root, 'labels' + postfix, '%s')

        subjects_list = []
        for image_name in sorted(listdir(images_folder)):
            if image_name in hold_out_val:
                label_name = get_label_name_from_image(image_name)
                num = image_name[4:7]
                subject = tio.Subject(
                    image = tio.ScalarImage(images % image_name),
                    label = tio.LabelMap(labels % label_name),
                    id = num 
                )
                subjects_list.append(subject)

        return tio.SubjectsDataset(subjects_list, tio.transforms.OneHot(num_classes=3))

    def train_dataloader(self):
        dataset = self.build_dataset(stage='train')
        sampler = Sampler(
            patch_size = self.patch_size,
            label_name = 'label',
            label_probabilities = {0:0, 1:0.2, 2:0.8}
        )
        queue = tio.Queue(
            dataset,
            max_length = 500,
            samples_per_volume = 10, # 27104 training slices, 162 case. 10 Samples per case requires 502 epochs (of my train_dataloader) to get each slice approximately 30 times as per paper.
            sampler = sampler,
            num_workers = 18
        )

        return DataLoader(queue, batch_size = 128, num_workers = 0)

    def val_dataloader(self):
        dataset = self.build_dataset('val')
        return DataLoader(dataset, batch_size=1, num_workers=6)

    def predict_dataloader(self):
        dataset = self.build_dataset(stage='test')
        return DataLoader(dataset, batch_size=1, num_workers=6)