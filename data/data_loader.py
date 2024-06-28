import os
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import ast
import cv2
import pydicom

import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer


from .constants import *
from .utils import get_imgs, read_from_dicom,resize_img

from albumentations import Compose, Normalize, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2

from .augmentations import BYOLAugmentations


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
###################################################################################################
# MULTI-MODAL (IMG & TEXT) DATA
###################################################################################################
class MultimodalPretrainingDataset(Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize    = imsize
        
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
            
        self.df.reset_index(drop=True, inplace=True)
        
        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(BASE_DIR, "captions.pickle")

        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii") #Removing any non-ASCII characters and ensuring the tokens are valid.
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
                                sent,
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                max_length=self.max_words,
                            )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs = [[] for _ in range(2)]
    cap_len, ids, tokens, attention = [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, p = b
        for i in range(2):
            imgs[i].append(img[i])
        
        # imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    # imgs = torch.stack(imgs)
    imgs = [torch.stack(image_list, dim=0) for image_list in imgs]
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": [img[sorted_cap_indices] for img in imgs],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict


###################################################################################################
# CLASSIFICATION DATA
###################################################################################################

class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
        

class MIMICImageDataset(BaseImageDataset):
    def __init__(self,split="train", transform=None, 
                 data_pct=1.0,img_type="Frontal",imsize=256,task = CHEXPERT_COMPETITION_TASKS):
        super().__init__(split, transform)
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(
                "MIMIC CXR data directory %s does not exist!" % MIMIC_CXR_DATA_DIR)
        self.imsize = imsize
        # read in csv file
        if split == "train":
            self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
        else:
            self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)
            
        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]
       
        # get a fraction of dataset
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        # get path
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))
        
        # fill na with 0s
        self.df = self.df.fillna(0) 
        # replace uncertains
        uncertain_mask = {k: -1 for k in task}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS) 
        
        self.path = self.df["Path"].values
        self.labels = self.df.loc[:, task].values
        
    def __getitem__(self, index):
        
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return x, y

    def __len__(self):
        return len(self.df)
    
    
class CheXpertImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 img_type="Frontal", data_pct=0.01, imsize=256,task =CHEXPERT_COMPETITION_TASKS):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")
        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(CHEXPERT_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(CHEXPERT_VALID_CSV)
        elif split == "test":
            self.df = pd.read_csv(CHEXPERT_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(
                CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in task}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.path = self.df["Path"].values
        self.labels = self.df.loc[:, task].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return x, y

    def __len__(self):
        return len(self.df)    
    
    
class NIHImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                  data_pct=0.01, imsize=256, task = NIH_TASKS):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(NIH_CXR_DATA_DIR):
            raise RuntimeError(f"{NIH_CXR_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(NIH_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(NIH_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        #get path
        self.df[NIH_PATH_COL] = self.df[NIH_PATH_COL].apply(lambda x: os.path.join(
                                    NIH_CXR_DATA_DIR, "/".join(x.split("/")[:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        self.path = self.df[NIH_PATH_COL].values
        self.labels = self.df.loc[:, task].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)

        return x, y

    def __len__(self):
        return len(self.df)    


class RSNAImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 phase="classification", data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(RSNA_TRAIN_CSV)
        elif self.split == "valid":
            self.df = pd.read_csv(RSNA_VALID_CSV)
        elif self.split == "test":
            self.df = pd.read_csv(RSNA_TEST_CSV)
        else:
            raise ValueError(f"split {split} does not exist!")

        if phase == "detection":
            self.df = self.df[self.df["Target"] == 1]

        self.df["Path"] = self.df["patientId"].apply(
            lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["Path"]
        x = read_from_dicom(
            img_path, self.imsize, self.transform)
        y = float(row["Target"])
        y = torch.tensor([y])

        return x, y
    
class COVIDXImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(COVIDX_DATA_DIR):
            raise RuntimeError(f"{COVIDX_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(COVIDX_TRAIN_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: COVIDX_DATA_DIR / f"train/{x}")
        elif self.split == "valid":
            self.df = pd.read_csv(COVIDX_VALID_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: COVIDX_DATA_DIR / f"train/{x}")
        elif self.split == "test":
            self.df = pd.read_csv(COVIDX_TEST_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: COVIDX_DATA_DIR / f"test/{x}")
        else:
            raise ValueError(f"split {split} does not exist!")

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["filename"]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = float(row["labels"])
        y = torch.tensor([y])
        return x, y
    
###################################################################################################
# SEGMENTATION DATA
###################################################################################################
class SIIMImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, phase="segmentation", imsize=224):
        super().__init__(split, transform)

        self.phase = phase
        self.imsize = imsize
        if self.phase == "segmentation":
            self.seg_transform = self.get_transforms()
        else:
            raise NotImplementedError(f"{self.phase} not implemented")

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(PNEUMOTHORAX_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(PNEUMOTHORAX_VALID_CSV)
        else:
            self.df = pd.read_csv(PNEUMOTHORAX_TEST_CSV)

        self.df["Path"] = self.df["Path"].apply(
            lambda x: os.path.join(PNEUMOTHORAX_IMG_DIR, x))

        # only keep positive samples for segmentation
        self.df["class"] = self.df[" EncodedPixels"].apply(lambda x: x != " -1")
        if self.phase == "segmentation" and split == "train":
            self.df_neg = self.df[self.df["class"] == False]
            self.df_pos = self.df[self.df["class"] == True]
            n_pos       = self.df_pos["ImageId"].nunique()
            neg_series  = self.df_neg["ImageId"].unique()
            
            neg_series_selected = np.random.choice(neg_series, size=n_pos, replace=False)
            self.df_neg = self.df_neg[self.df_neg["ImageId"].isin(neg_series_selected)]
            self.df = pd.concat([self.df_pos, self.df_neg])

        # sample data
        if data_pct != 1 and split == "train":
            ids = self.df["ImageId"].unique()
            n_samples = int(len(ids) * data_pct)
            series_selected = np.random.choice(ids, size=n_samples, replace=False)
            self.df = self.df[self.df["ImageId"].isin(series_selected)]

        self.imgids = self.df.ImageId.unique().tolist()

    def __getitem__(self, index):
        imgid = self.imgids[index]
        imgid_df = self.df.groupby("ImageId").get_group(imgid)

        # get image
        img_path = imgid_df.iloc[0]["Path"]
        x = self.read_from_dicom(img_path)

        # get labels
        if self.phase == "segmentation":
            rle_list = imgid_df[" EncodedPixels"].tolist()
            mask = np.zeros([1024, 1024])
            if rle_list[0] != " -1":
                for rle in rle_list:
                    mask += self.rle2mask(rle, PNEUMOTHORAX_IMG_SIZE, PNEUMOTHORAX_IMG_SIZE)
            mask = (mask >= 1).astype("float32")
            mask = resize_img(mask, self.imsize)

            augmented = self.seg_transform(image=x, mask=mask)
            x = augmented["image"]
            y = augmented["mask"].squeeze()
        else:
            y = imgid_df.iloc[0]["Label"]
            y = torch.tensor([y])

        return x, y

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def __len__(self):
        return len(self.imgids)

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def rle2mask(self, rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position: current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms


class RSNASegmentDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(RSNA_DETECTION_TRAIN_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(RSNA_DETECTION_VALID_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(RSNA_DETECTION_TEST_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        # self.df["Path"] = self.df["patientId"].apply(
        #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

        self.imsize = imsize
        self.seg_transform = self.get_transforms()

    def __len__(self):
        return len(self.filenames)

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = RSNA_IMG_DIR / filename
        x = self.read_from_dicom(img_path)

        mask = np.zeros([1024, 1024])

        bbox = self.bboxs[index]
        new_bbox = bbox[bbox[:, 3] > 0].astype(np.int64)
        if len(new_bbox) > 0:
            for i in range(len(new_bbox)):
                try:
                    mask[new_bbox[i, 1]:new_bbox[i, 3],
                         new_bbox[i, 0]:new_bbox[i, 2]] += 1
                except:
                    import ipdb
                    ipdb.set_trace()
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        augmented = self.seg_transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"].squeeze()

        return x, y

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms

###################################################################################################
# CONTEXT DATA
###################################################################################################
class ContextPretrainingDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,data_pct=1, task = NIH_TASKS):
        super().__init__(split=split, transform=transform)
        
        if not os.path.exists(NIH_CXR_DATA_DIR):
            raise RuntimeError(f"{NIH_CXR_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(NIH_TRAIN_BBOX_CSV)
        elif split == "valid":
            self.df = pd.read_csv(NIH_TEST_BBOX_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        #get path
        self.df[NIH_PATH_COL] = self.df[NIH_PATH_COL].apply(lambda x: os.path.join(
                                    NIH_CXR_DATA_DIR, "/".join(x.split("/")[:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        self.path = self.df[NIH_PATH_COL].values
        self.labels = self.df.loc[:, task].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = Image.open(img).convert('RGB')
        rlb = ast.literal_eval(self.df['Right_Lung_BBox'][index])
        llb = ast.literal_eval(self.df['Left_Lung_BBox'][index])
        timage = self.transform(x,rlb,llb )
        rlb, llb = self.resize_bbox(x, timage[0], rlb, llb)
        
        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return timage, y, rlb, llb

    def __len__(self):
        return len(self.df)    
    
    def resize_bbox(self,orig_image, resize_image, right_bbox,left_bbox):
    
        orig_image_size   = orig_image.size
        resize_image_size = resize_image.size()

        width_scale = resize_image_size[1] / orig_image_size[0]
        height_scale = resize_image_size[2] / orig_image_size[1]

        new_right_bbox = [
                    int(right_bbox[0] * width_scale),    # New x-coordinate of the top-left corner
                    int(right_bbox[1] * height_scale),   # New y-coordinate of the top-left corner
                    int(right_bbox[2] * width_scale),    # New x-coordinate of the bottom-right corner
                    int(right_bbox[3] * height_scale)    # New y-coordinate of the bottom-right corner
                ]
        new_left_bbox = [
                int(left_bbox[0] * width_scale),    # New x-coordinate of the top-left corner
                int(left_bbox[1] * height_scale),   # New y-coordinate of the top-left corner
                int(left_bbox[2] * width_scale),    # New x-coordinate of the bottom-right corner
                int(left_bbox[3] * height_scale)    # New y-coordinate of the bottom-right corner
            ]
        return new_right_bbox, new_left_bbox
    
def context_collate_fn(batch):
    images = [[] for _ in range(4)]
    labels = []
    rbboxes, lbboxes = [], []

    for item in batch:
        for i in range(4):
            images[i].append(item[0][i])
        labels.append(item[1])
        if len(item) > 2:
            rbboxes.append(item[2])
            lbboxes.append(item[3])

    images = [torch.stack(image_list, dim=0) for image_list in images]
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)

    return images, labels, rbboxes, lbboxes  
    

        
class DataLoader():
    def __init__(self, config=None):
        self.config = config
        self.pre_bs          = config['pre_bs']
        self.train_bs        = config['train_bs']
        self.val_bs          = config['val_bs']
        self.data_workers    = config['data_workers']
        self.data_pct        = config['data_pct']/100.0
         
        self.multimodalcollate = multimodal_collate_fn
        self.contextcollate    = context_collate_fn
        
        if config['dataset'] == 'nih':
            if config['task'] == 'nih_tasks':
                self.task = NIH_TASKS
            if config['task'] == 'nih_cxr8_tasks':
                self.task = NIH_CXR8_TASKS        
        
        if config['dataset'] == 'mimic' or config['dataset'] == 'chexpert':
            if config['task'] == 'chex_competiton_tasks':
                self.task = CHEXPERT_COMPETITION_TASKS
            if config['task'] == 'chex_tasks':
                self.task = CHEXPERT_TASKS
              
        if config['mode'] == 'down':    
            self.train_transform = T.Compose([T.Resize((224, 224)), 
                                              T.RandomHorizontalFlip(),
                                              T.transforms.RandomGrayscale(p=0.2), 
                                              T.ToTensor(),           
                                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                            ])
                
        elif config['mode'] == 'pre':
            self.train_transform = BYOLAugmentations(self.config)

        self.valid_augmentations = T.Compose([T.Resize((224, 224)), 
                                                  T.RandomHorizontalFlip(),
                                                  T.ToTensor(),           
                                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                                ])
        
    #####################################################################
    # CLASSIFICATION LOADERS (For Downstream Tasks)
    #####################################################################
    
    def GetMimicDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = MIMICImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct = self.data_pct,
                                      task = self.task
                               )
        valid_set = MIMICImageDataset(split="valid",
                                 transform =valid_transform,
                                 task = self.task
                               )
        test_set = MIMICImageDataset(split="test",
                               transform =valid_transform,
                               task = self.task
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=64,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=64,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')
        
        return train_loader, valid_loader, test_loader
    
    def GetChexpertDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = CheXpertImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                                      task = self.task   
                               )
        valid_set = CheXpertImageDataset(split="valid",
                                 transform =valid_transform,
                                 task = self.task
                               )
        test_set  = CheXpertImageDataset(split="test",
                               transform =valid_transform,
                               task = self.task
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')

        nih
        return train_loader, valid_loader, test_loader
    
    def GetNihDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = NIHImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                                      task = self.task
                               )
        valid_set = NIHImageDataset(split="valid",
                                 transform =valid_transform,
                                 task = self.task
                               )
        
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
                
        return train_loader, valid_loader, valid_loader
    
    def GetCovidxDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = COVIDXImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct
                               )
        valid_set = COVIDXImageDataset(split="valid",
                                 transform =valid_transform
                               )
        test_set  = COVIDXImageDataset(split="test",
                               transform =valid_transform
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')

        
        return train_loader, valid_loader, test_loader
    
    def GetRsnaDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = RSNAImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct
                               )
        valid_set = RSNAImageDataset(split="valid",
                                 transform =valid_transform
                               )
        test_set  = RSNAImageDataset(split="test",
                               transform =valid_transform
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')

        
        return train_loader, valid_loader, test_loader
    
    #####################################################################
    # SEGMENTATION LOADERS (For Downstream Tasks)
    #####################################################################
    
    def GetSIIMDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = SIIMImageDataset(split="train",
                                      transform = None,
                                      data_pct=self.data_pct
                               )
        valid_set = SIIMImageDataset(split="valid",
                                 transform =None
                               )
        test_set  = SIIMImageDataset(split="test",
                               transform =None
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')

        return train_loader, valid_loader, test_loader
    
    
    def GetRSNASegmentDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = RSNASegmentDataset(split="train",
                                      transform = None,
                                      data_pct=self.data_pct
                               )
        valid_set = RSNASegmentDataset(split="valid",
                                 transform =None
                               )
        test_set  = RSNASegmentDataset(split="test",
                               transform =None
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')
        
        return train_loader, valid_loader, test_loader
    
    #####################################################################
    # CONTEXT AND MULTIMODAL LOADERS (For Pre-training Tasks)
    ##################################################################### 
       
    def GetContextPretrainingDataset(self):            
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations
        
        train_set = ContextPretrainingDataset(split="train",
                                              transform = train_transform,
                                               )
        
        valid_set = ContextPretrainingDataset(split="valid",
                                              transform =valid_transform
                                              )
        train_loader = DL(dataset=train_set,
                         batch_size=self.pre_bs,
                         collate_fn=self.collate_fn,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.pre_bs,
                         collate_fn=self.collate_fn,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        
        return train_loader, valid_loader
    
    
    def GetMultimodalPretrainingDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = MultimodalPretrainingDataset(split="train",
                                      transform = train_transform,
                               )
        valid_set = MultimodalPretrainingDataset(split="valid",
                                 transform =valid_transform
                               )
        test_set = MultimodalPretrainingDataset(split="test",
                               transform =valid_transform
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.pre_bs,
                         collate_fn = self.multimodalcollate,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.pre_bs,
                         collate_fn = self.multimodalcollate,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
       
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        
        return train_loader, valid_loader
            
