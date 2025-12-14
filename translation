import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# 模型设置
# =========================
ZH_EN_MODEL = "Helsinki-NLP/opus-mt-zh-en"
EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"

# 固定版本（用于可复现性）
REVISION = "cf109095479db38d6df799875e34039d4938aaa6"

# 加载模型与 tokenizer
tokenizer_zh_en = AutoTokenizer.from_pretrained(
    ZH_EN_MODEL,
    revision=REVISION
)
model_zh_en = AutoModelForSeq2SeqLM.from_pretrained(
    ZH_EN_MODEL,
    revision=REVISION
)

tokenizer_en_zh = AutoTokenizer.from_pretrained(EN_ZH_MODEL)
model_en_zh = AutoModelForSeq2SeqLM.from_pretrained(EN_ZH_MODEL)


# =========================
# 判断是否包含中文
# =========================
def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


# =========================
# 单句翻译函数
# 返回：
#   src_lang: "ZH" or "EN"
#   src_text: 原文
#   tgt_text: 译文
# =========================
def translate(text: str):
    if contains_chinese(text):
        # 中文 → 英文
        inputs = tokenizer_zh_en(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        outputs = model_zh_en.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=256
        )
        en_text = tokenizer_zh_en.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return "ZH", text, en_text

    else:
        # 英文 → 中文
        inputs = tokenizer_en_zh(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        outputs = model_en_zh.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=256
        )
        zh_text = tokenizer_en_zh.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return "EN", text, zh_text


# =========================
# 文件级翻译
# 生成：
#   1) 双语对照文件
#   2) 纯目标语文件
# =========================
def translate_file(
    input_path: str,
    bilingual_path: str,
    target_only_path: str
):
    # 读取输入
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 打开两个输出文件
    with open(bilingual_path, "w", encoding="utf-8") as f_bi, \
         open(target_only_path, "w", encoding="utf-8") as f_tgt:

        for idx, line in enumerate(lines, start=1):
            try:
                src_lang, src, tgt = translate(line)

                if src_lang == "ZH":
                    # ===== 双语文件 =====
                    f_bi.write(f"[{idx}]\n")
                    f_bi.write("[ZH]\n")
                    f_bi.write(src + "\n")
                    f_bi.write("[EN]\n")
                    f_bi.write(tgt + "\n\n")

                    # ===== 纯目标语文件 =====
                    f_tgt.write(tgt + "\n")

                else:
                    # ===== 双语文件 =====
                    f_bi.write(f"[{idx}]\n")
                    f_bi.write("[EN]\n")
                    f_bi.write(src + "\n")
                    f_bi.write("[ZH]\n")
                    f_bi.write(tgt + "\n\n")

                    # ===== 纯目标语文件 =====
                    f_tgt.write(tgt + "\n")

                print(f"[{idx}/{len(lines)}] OK")

            except Exception as e:
                print(f"[{idx}] ERROR: {e}")
                f_bi.write(f"[{idx}]\n")
                f_bi.write("[ERROR]\n")
                f_bi.write(line + "\n\n")


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    INPUT_FILE = "aligned_input.txt"

    BILINGUAL_FILE = "bilingual.txt"
    TARGET_ONLY_FILE = "target_only.txt"

    translate_file(
        INPUT_FILE,
        BILINGUAL_FILE,
        TARGET_ONLY_FILE
    )

    print("Finished.")
