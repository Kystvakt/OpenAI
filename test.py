import openai
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    prog="TEST",
    usage="%(prog)s [options]",
    description="Prompt settings",
)
parser.add_argument("-n", dest="n", default=1, type=int, metavar="N", help="required number of responses")
parser.add_argument(
    "-c", "--content",
    nargs="?",
    const="contents/injection.txt",
    default="contents/default.txt",
    metavar="file_path",
    help="content file path"
)
args = parser.parse_args()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Creating prompt
with open(args.content, "r", encoding="UTF-8") as f:
    content = f.read()


guideline = "당신은 심리 상담사입니다. 다음 일기를 읽고, 다섯 문장 이내로 구체적인 지시를 담은 질문을 합니다.\n\n"
prompt = guideline + "일기: " + content + "\n반응:"

# Create completions
PARAMS = {
    "model": "text-davinci-003",
    "prompt": prompt,
    "suffix": "\n",
    "max_tokens": 500,
    "top_p": 0.8,
    "n": args.n,
}
print("프롬프트:", guideline)
print("일기:", content)
response = openai.Completion.create(**PARAMS)

print("\n반응:")
for idx in range(len(response["choices"])):
    print(f"[{idx + 1}] " + response["choices"][idx]["text"])
