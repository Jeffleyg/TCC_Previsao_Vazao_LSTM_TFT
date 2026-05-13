import torch

print(f"PyTorch versão: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memória Total: {torch.cuda.get_get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ GPU não detectada. O treino será feito via CPU (Lento).")