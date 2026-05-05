from src.TANTGen.TANTGen_NIDS import TANTGen_NIDS
from src.TANTGen.generate_sequences import generate_sliding_windows_NIDS
from src.TANTGen.utils import PacketDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

sliding_window_size = 64
window_stride = 64
num_features = 2
batch_size = 64
dataset_path = "data/preprocessed/adversarial_test/Mirai_malicious.pcap"
dataset_features_path = "data/preprocessed/adversarial_test/Mirai_malicious_features.csv"
# dataset_features_path = "data/adversarial/Mirai_adversarial_features.csv"
#dataset_labels_path = "data/preprocessed/adversarial_test/Mirai_labels.csv"

#dataset_path = "data/preprocessed/adversarial_test/Active_Wiretap_malicious.pcap"
dataset_features_path = "data/preprocessed/adversarial_test/Active_Wiretap_malicious_features.csv"
dataset_labels_path = "data/adversarial/temp_malicious_labels.csv"

# dataset_path = "data/adversarial/temp_malicious.pcap"
# dataset_features_path = "data/adversarial/temp_features.csv"
# dataset_labels_path = "data/adversarial/temp_malicious_labels.csv"

# df = get_basic_features(dataset_path)
# df.to_csv(dataset_features_path, index=False)

feature_arr, target_arr = generate_sliding_windows_NIDS(
    sliding_window_size,
    window_stride,
    dataset_features_path,
    dataset_labels_path,
    "src/TANTGen/models/mean_std.json" # TODO
)

test_dataset = PacketDataset(feature_arr, target_arr)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

callbacks = [
    EarlyStopping(monitor="val_f1", mode="max", patience=6),
    ModelCheckpoint(monitor="val_f1", mode="max", filename="best_nids_{epoch:02d}-{val_f1:.4f}")
]

if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print("Number of GPUS: {}".format(NUM_GPUS))
else:
    print("ERROR: NO CUDA DEVICE FOUND")
    NUM_GPUS = 0

if NUM_GPUS >= 1:
    trainer = pl.Trainer(
        precision=32,
        accelerator="gpu",
        devices=1,
        max_epochs=15,
        callbacks=callbacks,
    )
else:
    trainer = pl.Trainer(
        max_epochs=15,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

cpath = "{}/{}{}.ckpt".format(
            "src/TANTGen/transformer_nids_logs", "TANTGen_NIDS_window", sliding_window_size
        )

testmodel = TANTGen_NIDS.load_from_checkpoint(
    freeze_base=True,
    unfreeze_last_n_layers=0,
    checkpoint_path=cpath,
    strict=True,
)
testmodel.eval()

trainer.test(testmodel, dataloaders=test_loader)