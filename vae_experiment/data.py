from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import device, batch_size, class_idx

#############################
#############################
# Датасеты
#############################
#############################

transform = transforms.Compose([
    transforms.ToTensor(),   # [0,1]
])

root = './data'
train_all = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
test_all  = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

# Фильтруем: train содержит только класс со Свитерами
train_idxs = [i for i, (_, y) in enumerate(train_all) if y == class_idx]
train_ds = Subset(train_all, train_idxs)

# Для валидации/теста подготовим смесь: нормальные (class_idx) + аномальные (все остальные)
def make_eval_subset(dataset, include_all_other_classes=True, max_per_class=None):
    idxs_normal = [i for i, (_, y) in enumerate(dataset) if y == class_idx]
    idxs_anom = [i for i, (_, y) in enumerate(dataset) if y != class_idx]
    if max_per_class is not None:
        # ограничение: взять равное количество из аномалий
        idxs_anom = idxs_anom[:max_per_class]
    idxs = idxs_normal + idxs_anom
    return Subset(dataset, idxs), len(idxs_normal), len(idxs_anom)

val_ds, n_norm_val, n_anom_val = make_eval_subset(test_all, max_per_class=1000)
test_ds, n_norm_test, n_anom_test = make_eval_subset(test_all, max_per_class=None)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

print("Train size (normal only):", len(train_ds))
print("Val size (normal/anom):", len(val_ds), f"({n_norm_val} normal, {n_anom_val} anom)")
print("Test size (normal/anom):", len(test_ds), f"({n_norm_test} normal, {n_anom_test} anom)")
