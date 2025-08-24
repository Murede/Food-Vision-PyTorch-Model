# Inference script

      # 9. Prepare model for inference by sending it to target device and turning one eval() mode
      with torch.inference_mode():
        pred_logit = model(transformed_image) # perform inference on target sample
        with torch.inference_mode():
plt.savefig("images/09-foodvision-mini-inference-speed-vs-performance.jpg")
def predict(img) -> Tuple[Dict, float]:
  # Put model into evaluation mode and turn on inference mode
  with torch.inference_mode():
def predict(img) -> Tuple[Dict, float]:
    # Put model into evaluation mode and turn on inference mode
    with torch.inference_mode():
print(f"Testing/inference transforms: {effnetb2_transforms}")
