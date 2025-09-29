# NSGA-Net GPU Setup Instructions

## ✅ Changes Made for GPU Compatibility

### 1. **Fixed Path Issues**

- Updated all `sys.path.insert()` statements to use your actual path: `r'D:\Research\nsga-net'`
- Files updated:
  - `search/evolution_search.py`
  - `search/train_search.py`
  - `validation/train.py`
  - `validation/test.py`

### 2. **Fixed GPU Device Detection**

- Changed hardcoded `device = 'cuda'` to `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- Added proper CUDA initialization checks
- Files updated:
  - `search/train_search.py`
  - `validation/train.py`
  - `validation/test.py`

### 3. **Fixed CUDA Operations**

- Replaced deprecated `.cuda()` calls with `.to(device)`
- Added proper CUDA seed initialization
- Fixed GPU memory management

### 4. **Fixed Model Issues**

- Added missing `droprate` attribute to all network classes
- Fixed numpy compatibility issues

### 5. **Created Helper Scripts**

- `test_gpu_setup.py` - Test GPU setup and imports
- `run_evolution_search.py` - Easy script to run evolution search
- `sitecustomize.py` - NumPy compatibility fixes

## 🚀 How to Run NSGA-Net

### **Option 1: Quick Test (Recommended first)**

```bash
python run_evolution_search.py --quick
```

This runs a minimal test with 4 architectures, 2 generations, and 5 epochs each.

### **Option 2: Full Evolution Search**

```bash
# Micro search space (recommended)
python run_evolution_search.py --search_space micro --pop_size 20 --n_gens 10 --epochs 20

# Macro search space
python run_evolution_search.py --search_space macro --pop_size 20 --n_gens 10 --epochs 20
```

### **Option 3: Direct Command**

```bash
python search/evolution_search.py --search_space micro --pop_size 20 --n_gens 10 --epochs 20 --init_channels 24 --layers 11
```

## 📊 Your System Specs

- **GPU**: NVIDIA GeForce GTX 1650
- **CUDA**: 12.1
- **PyTorch**: 2.5.1+cu121
- **Status**: ✅ Ready to run

## ⚙️ Parameter Recommendations

### **For Your GTX 1650 (4GB VRAM):**

**Conservative (Safe):**

```bash
python run_evolution_search.py --search_space micro --pop_size 10 --n_gens 5 --epochs 15 --init_channels 16
```

**Moderate:**

```bash
python run_evolution_search.py --search_space micro --pop_size 20 --n_gens 10 --epochs 20 --init_channels 24
```

**Aggressive (Monitor GPU memory):**

```bash
python run_evolution_search.py --search_space micro --pop_size 30 --n_gens 15 --epochs 25 --init_channels 32
```

## 🔍 Monitoring GPU Usage

While training, monitor your GPU usage with:

```bash
# In another terminal
nvidia-smi -l 1
```

## 📁 Output Files

Results will be saved in:

- `search-{experiment_name}-{search_space}-{timestamp}/`
- Each architecture gets its own folder: `arch_1/`, `arch_2/`, etc.
- Logs and model checkpoints are saved automatically

## 🐛 Troubleshooting

### **If you get CUDA out of memory:**

1. Reduce `--init_channels` (try 16 instead of 24)
2. Reduce `--pop_size`
3. Reduce `--epochs`

### **If training is too slow:**

1. Reduce `--epochs` (try 10-15)
2. Reduce `--pop_size`
3. Use `--search_space micro` (faster than macro)

### **If you want to test first:**

```bash
python test_gpu_setup.py
```

## 🎯 Expected Results

- **Micro search space**: Typically finds architectures with 2-4% error rate on CIFAR-10
- **Macro search space**: More diverse architectures, may take longer to converge
- **Training time**: ~30-60 minutes per generation depending on parameters

## 📝 Notes

- The code will automatically download CIFAR-10 dataset on first run
- GPU memory usage will vary based on architecture complexity
- You can stop training anytime with Ctrl+C
- Results are saved incrementally, so you won't lose progress

---

**Ready to start? Run:**

```bash
python run_evolution_search.py --quick
```
