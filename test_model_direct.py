#!/usr/bin/env python3
"""
Direct test of the trained model to see what it actually outputs.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
model_path = "./checkpoints/llama_lrs3_full"
base_model = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, model_path)
model.eval()

print("\n" + "="*60)
print("Testing with training format examples")
print("="*60)

# Test 1: Simple example
test_cases = [
    {
        "phonemes": "HH AH L OW",
        "expected": "HELLO"
    },
    {
        "phonemes": "W ER L D",
        "expected": "WORLD"
    },
    {
        "phonemes": "DH AH B OY IH Z K L AY M IH NG DH AH S T EH R Z",
        "expected": "THE BOY IS CLIMBING THE STAIRS"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}:")
    print(f"Phonemes: {test['phonemes']}")
    print(f"Expected: {test['expected']}")
    
    # Format EXACTLY as training data
    prompt = (
        "<PHONEMES>\n"
        "<PHONEME_SEQUENCE>\n"
        f"{test['phonemes']}\n"
        "</PHONEME_SEQUENCE>\n"
        "<TEXT>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract generated text
    if "<TEXT>" in full_output:
        generated = full_output.split("<TEXT>")[-1].strip()
        # Clean up
        for tag in ["</S2S>", tokenizer.eos_token, "\n"]:
            if tag and tag in generated:
                generated = generated.split(tag)[0]
        generated = generated.strip()
    else:
        generated = full_output
    
    print(f"Generated: {generated}")
    print(f"Match: {'✓' if generated == test['expected'] else '✗'}")

print("\n" + "="*60)
