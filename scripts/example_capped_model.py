from assistant_axes import CappedModel
from assistant_axes.contrastive import format_prompt


def main():
    print("Loading Qwen with default capping settings...")
    capped = CappedModel.from_model_key("qwen")

    prompt = format_prompt(
        system="You are a chronic contrarian who disagrees with everything.",
        query="What is 2+2?",
        model_type="qwen",
    )

    print("\n=== Without capping ===")
    uncapped_output = capped.generate_uncapped(prompt, max_new_tokens=100)
    response_start = uncapped_output.find("<|im_start|>assistant") + len("<|im_start|>assistant\n")
    print(uncapped_output[response_start:].strip())

    print("\n=== With capping ===")
    capped_output = capped.generate(prompt, max_new_tokens=100)
    print(capped_output[response_start:].strip())


if __name__ == "__main__":
    main()
