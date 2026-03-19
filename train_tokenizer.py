import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# 1. Create empty BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token=None))

logging.basicConfig(level=logging.INFO)

# 2. Use byte-level pre-tokenizer (GPT-style)
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

tokenizer.decoder = ByteLevelDecoder()

# 3. Setup trainer
trainer = BpeTrainer(
    vocab_size=20000,                 # your desired vocab size
    special_tokens = ["<|endoftext|>"],
    show_progress=True   # IMPORTANT
)

# 4. Train on your text files
files = ["/kaggle/input/fineweb-edu-500m-tokens/fineweb_edu_slice_500M.txt"]  # put your training text file here
tokenizer.train(files, trainer)

# 5. Save tokenizer
tokenizer.save("/kaggle/working/my_gpt_tokenizer.json")

print("Training complete!")
