# Data

This project uses a subset of the **BANKING77** dataset for fine-grained online-banking intent classification.

## Source

- Hugging Face: https://huggingface.co/datasets/PolyAI/banking77
- Original repository: https://github.com/PolyAI-LDN/task-specific-datasets
- Train CSV: https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/banking_data/train.csv
- Test CSV: https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/banking_data/test.csv
- Research paper: https://arxiv.org/abs/2003.04807

## License

CC BY 4.0

## Files in this folder

| File | Description |
|------|-------------|
| `train.csv` | 8-intent subset of BANKING77 training split (1,183 examples) |
| `test.csv` | 8-intent subset of BANKING77 test split (320 examples) |

## Selected Intents (8)

The files are pre-filtered to the **8 recommended intents**:

1. `card_arrival`
2. `card_not_working`
3. `cash_withdrawal_not_recognised`
4. `declined_card_payment`
5. `lost_or_stolen_card`
6. `transaction_charged_twice`
7. `transfer_not_received_by_recipient`
8. `cash_withdrawal_charge`

The original 10,003 / 3,080 train/test assignment is preserved — the subset is a filter, not a re-split. The column `category` holds the intent label and `text` holds the message.
