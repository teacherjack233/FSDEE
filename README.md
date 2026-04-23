# FSDSE: Fully Spiking Dynamic Expert Ensembles for Task-Free Continual Learning

## Requirements

```bash
pip install torch numpy matplotlib tqdm pandas scikit-learn
```

## Usage

### Basic Usage

```bash
python main.py --dataset mnist --num_tasks 5 --gpu 0
```

### Arguments

- `--dataset`: Dataset to use (default: `mnist`). Options: `mnist`, `cifar10`, `cifar100`, `permuted_mnist`
- `--num_tasks`: Number of tasks (default: `5`)
- `--gpu`: GPU ID (default: `0`)
- `--batch_size`: Batch size (default: `64`)
- `--n_epochs`: Number of training epochs per component (default: `20`)
- `--strategy`: Sample selection strategy (default: `sliding_window`). Options: `sliding_window`, `diversity`
- `--threshold`: Expansion threshold (default: `0.15`)
- `--n_steps`: Number of time steps (default: `8`)
- `--dataset_fraction`: Dataset sampling fraction (default: `0.5`)
- `--memory_size`: Memory buffer size (default: `2000`)
- `--save_dir`: Save directory (default: `results`)

### Examples

Train on MNIST with 5 tasks:
```bash
python main.py --dataset mnist --num_tasks 5 --gpu 0
```

Train on CIFAR-10 with 10 tasks:
```bash
python main.py --dataset cifar10 --num_tasks 10 --gpu 0 --n_epochs 30
```

Train on Permuted MNIST with diversity strategy:
```bash
python main.py --dataset permuted_mnist --num_tasks 10 --strategy diversity --gpu 0
```
