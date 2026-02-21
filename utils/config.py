# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "../../dataset/SAMUS/" 
    data_subpath = "../../dataset/SAMUS/ThyroidNodule-TN3K/" 
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/Echocardiography-CAMUS/" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_BUSI_Ext:
    data_path = "./data/processed"
    data_subpath = "./data/processed/Breast-BUSI-Ext/"
    save_path = "./checkpoints/BUSI_EXT/"
    result_path = "./result/BUSI_EXT/"
    tensorboard_path = "./tensorboard/BUSI_EXT/"
    load_path = "./checkpoints/SAMUS.pth"
    save_path_code = "_"

    workers = 1
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-Breast-BUSI-Ext"
    val_split = "val-Breast-BUSI-Ext"
    test_split = "test-Breast-BUSI-Ext"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUS_Ext:
    data_path = "./data/processed"
    data_subpath = "./data/processed/Breast-BUS-Ext/"
    save_path = "./checkpoints/BUS_EXT/"
    result_path = "./result/BUS_EXT/"
    tensorboard_path = "./tensorboard/BUS_EXT/"
    load_path = "./checkpoints/SAMUS.pth"
    save_path_code = "_"

    workers = 1
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-Breast-BUS-Ext"
    val_split = "val-Breast-BUS-Ext"
    test_split = "test-Breast-BUS-Ext"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSBRA_Ext:
    data_path = "./data/processed"
    data_subpath = "./data/processed/Breast-BUSBRA-Ext/"
    save_path = "./checkpoints/BUSBRA_EXT/"
    result_path = "./result/BUSBRA_EXT/"
    tensorboard_path = "./tensorboard/BUSBRA_EXT/"
    load_path = "./checkpoints/SAMUS.pth"
    save_path_code = "_"

    workers = 1
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-Breast-BUSBRA-Ext"
    val_split = "val-Breast-BUSBRA-Ext"
    test_split = "test-Breast-BUSBRA-Ext"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_US30K_NoCamus:
    # US30K with CAMUS excluded -- for AutoSAMUS foundation-model training.
    # CAMUS has multi-class masks (LV/MYO/LA) for the same image; AutoSAMUS
    # cannot disambiguate without a manual prompt, so we exclude it.
    # Run prepare_us30k_no_camus.py first to generate the filtered split files.
    data_path = "./US30K"
    save_path = "./checkpoints/US30K_NOCAMUS/"
    result_path = "./result/US30K_NOCAMUS/"
    tensorboard_path = "./tensorboard/US30K_NOCAMUS/"
    load_path = "./checkpoints/SAMUS.pth"
    save_path_code = "_"

    workers = 1
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train_no_camus"
    val_split = "val_no_camus"
    test_split = "test_no_camus"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_AllBreast:
    # Combined training on all 5 breast ultrasound datasets
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/AllBreast/"
    result_path = "./result/AllBreast/"
    tensorboard_path = "./tensorboard/AllBreast/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 5e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-AllBreast"
    val_split = "val-AllBreast"
    test_split = "val-AllBreast"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastBUSI:
    # BUSI dataset only (for per-dataset training/eval)
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/BreastBUSI/"
    result_path = "./result/BreastBUSI/"
    tensorboard_path = "./tensorboard/BreastBUSI/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-BUSI"
    val_split = "val-BUSI"
    test_split = "val-BUSI"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastBUSBRA:
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/BreastBUSBRA/"
    result_path = "./result/BreastBUSBRA/"
    tensorboard_path = "./tensorboard/BreastBUSBRA/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-BUSBRA"
    val_split = "val-BUSBRA"
    test_split = "val-BUSBRA"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastBUS:
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/BreastBUS/"
    result_path = "./result/BreastBUS/"
    tensorboard_path = "./tensorboard/BreastBUS/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-BUS"
    val_split = "val-BUS"
    test_split = "val-BUS"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastBUS_UC:
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/BreastBUS_UC/"
    result_path = "./result/BreastBUS_UC/"
    tensorboard_path = "./tensorboard/BreastBUS_UC/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-BUS_UC"
    val_split = "val-BUS_UC"
    test_split = "val-BUS_UC"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastBUS_UCLM:
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/BreastBUS_UCLM/"
    result_path = "./result/BreastBUS_UCLM/"
    tensorboard_path = "./tensorboard/BreastBUS_UCLM/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-BUS_UCLM"
    val_split = "val-BUS_UCLM"
    test_split = "val-BUS_UCLM"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastUDIAT:
    data_path = "./SAMUS_DATA/"
    save_path = "./checkpoints/BreastUDIAT/"
    result_path = "./result/BreastUDIAT/"
    tensorboard_path = "./tensorboard/BreastUDIAT/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-UDIAT"
    val_split = "val-UDIAT"
    test_split = "val-UDIAT"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

class Config_BreastBUSI_BUSBRA:
    # Combined BUSI + BUSBRA training (for baseline AutoSAMUS comparison)
    # Split files: train-BUSI_BUSBRA.txt, val-BUSI_BUSBRA.txt
    # Generated by concatenating per-dataset splits
    data_path = "./SAMUS/SAMUS_DATA/"
    save_path = "./checkpoints/BreastBUSI_BUSBRA/"
    result_path = "./result/BreastBUSI_BUSBRA/"
    tensorboard_path = "./tensorboard/BreastBUSI_BUSBRA/"
    load_path = "./checkpoints/samus_pretrained.pth"
    save_path_code = "_"

    workers = 1
    epochs = 200
    batch_size = 8
    learning_rate = 5e-4
    momentum = 0.9
    classes = 2
    img_size = 256
    train_split = "train-BUSI_BUSBRA"
    val_split = "val-BUSI_BUSBRA"
    test_split = "val-BUSI_BUSBRA"
    crop = None
    eval_freq = 1
    save_freq = 2000
    device = "cuda"
    cuda = "on"
    gray = "yes"
    img_channel = 1
    eval_mode = "mask_slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"

# ==================================================================================================
def get_config(task="US30K"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "BUSI_EXT":
        return Config_BUSI_Ext()
    elif task == "BUS_EXT":
        return Config_BUS_Ext()
    elif task == "BUSBRA_EXT":
        return Config_BUSBRA_Ext()
    elif task == "US30K_NOCAMUS":
        return Config_US30K_NoCamus()
    elif task == "AllBreast":
        return Config_AllBreast()
    elif task == "BreastBUSI":
        return Config_BreastBUSI()
    elif task == "BreastBUSBRA":
        return Config_BreastBUSBRA()
    elif task == "BreastBUS":
        return Config_BreastBUS()
    elif task == "BreastBUS_UC":
        return Config_BreastBUS_UC()
    elif task == "BreastBUS_UCLM":
        return Config_BreastBUS_UCLM()
    elif task == "BreastUDIAT":
        return Config_BreastUDIAT()
    elif task == "BreastBUSI_BUSBRA":
        return Config_BreastBUSI_BUSBRA()
    else:
        assert("We do not have the related dataset, please choose another task.")