import torchvision
import scipy.io

class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content

class DOGDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

 

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0] for o in image_info]
        
        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0] for o in split_info]
        self.split_info = {}
        if split == 'train' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split== 'test' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"
                    
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

 

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file = self.is_valid_file,
                                         *args, **kwargs)

 

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

 

    @staticmethod
    def get_file_content(file_path):
        content =  scipy.io.loadmat(file_path)
        return content['file_list']


if __name__ == '__main__':
    data_root = "/home/muzammalnaseer/fahad_labs/cub/CUB_200_2011"
    transform = torchvision.transforms.ToTensor()
    train_dataset = CUBDataset(image_root_path=f"{data_root}", transform=transform, split="train")
    test_dataset = CUBDataset(image_root_path=f"{data_root}", transform=transform, split="test")
