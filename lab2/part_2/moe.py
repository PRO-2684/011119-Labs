import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tokenizer:
    def __init__(self, dataPath: str):
        with open(dataPath, "r", encoding="utf-8") as f:
            self.dataset = f.read()
        self.generate_vocabulary()

    def generate_vocabulary(
        self,
    ) -> None:
        # FIXME:
        # Create a sorted list of unique characters
        unique_chars = sorted(set(self.dataset))
        # Create a dictionary to map each character to a unique index
        self.char2index = {char: idx + 1 for idx, char in enumerate(unique_chars)}
        self.char2index["<START>"] = 0  # Adding start token
        self.char2index["<END>"] = len(self.char2index)  # Adding end token
        # Create a dictionary to map each index back to its character
        self.index2char = {idx: char for char, idx in self.char2index.items()}


    def encode(
        self,
        sentence: str,
    ) -> torch.Tensor:
        """
        # FIXME:
        例子, 假设 A-Z 对应的 token 是 1-26, 句子开始，结束符号的 token 是 0。
        input  : "ABCD"
        output : Tensor([0,1,2,3])

        注意: 为了后续实验方便，输出 Tensor的 数据类型 dtype 为 torch.long。
        """
        # Encode the sentence to a list of indices
        tokens = [self.char2index["<START>"]] + [self.char2index[char] for char in sentence] + [self.char2index["<END>"]]
        # Convert the list to a tensor of type torch.long
        return torch.tensor(tokens, dtype=torch.long)


    def decode(
        self,
        tokens: torch.Tensor,
    ) -> str:
        """
        # FIXME:
        例子, 假设 A-Z 对应的 token 是 1-26, 句子开始，结束符号的 token 是 0。
        input : Tensor([0,1,2,3])
        output : "ABCD"
        """
        # Decode the tensor of indices back to a string
        chars = [self.index2char[idx.item()] for idx in tokens if idx.item() in self.index2char]
        # Join the characters to form the decoded string
        return "".join(chars).replace("<START>", "").replace("<END>", "")


class ShakespeareDataset(Dataset):
    def __init__(self, filepath, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        self.encoded = self.tokenizer.encode(text)
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.encoded) - self.chunk_size

    def __getitem__(self, idx):
        # FIXME:提取一段文本(长度为 chunk_size）作为输入，以及这段文本的每一个字符的下一个字符作为标签
        # example(not correspond to real text): chunk = tensor([ 0, 20, 49, 58, 59])
        #         label = tensor([20, 49, 58, 59, 19])
        # decoded chunk: "The "
        # decoded label: "he T"
        chunk = self.encoded[idx:idx + self.chunk_size]
        label = self.encoded[idx + 1:idx + self.chunk_size + 1]
        return chunk, label


tokenizer = Tokenizer(dataPath="input.txt")


def create_dataloader(filepath, tokenizer, chunk_size, batch_size, shuffle=True):
    dataset = ShakespeareDataset(filepath, tokenizer, chunk_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader


class HeadAttention(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, hidden_size: int):
        super().__init__()
        # embed_size: dimension for input embedding vector
        # hidden_size: dimension for hidden vector. eg. x:(..., embed_size) --to_q--> query_vector:(..., hidden_size)

        # a triangular bool matrix for mask
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

        # FIXME:
        # Initialize three linear transformations for Q, K, and V
        self.to_q = nn.Linear(embed_size, hidden_size)
        self.to_k = nn.Linear(embed_size, hidden_size)
        self.to_v = nn.Linear(embed_size, hidden_size)


    def forward(self, inputs) -> torch.Tensor:
        # input: (batch_size, seq_len, embed_size)
        # return (batch_size, seq_len, hidden_size)

        # FIXME:
        batch_size, seq_len, embed_size = inputs.shape

        # Apply linear transformations to get Q, K, V
        Q = self.to_q(inputs)  # (batch_size, seq_len, hidden_size)
        K = self.to_k(inputs)  # (batch_size, seq_len, hidden_size)
        V = self.to_v(inputs)  # (batch_size, seq_len, hidden_size)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scores = scores / (embed_size ** 0.5)  # Scale scores

        # Apply mask
        mask = self.tril[:seq_len, :seq_len]  # Ensure mask has the same size as scores
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Multiply attention weights with V
        out = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_size)

        return out


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention is consist of many HeadAttention output.
    # concat all this head attention output o_i, then merge them with a projection matrix W_o, as [o_1, o_2, ...] x W_o
    # The reason for using multi-head attention is that we want each head to be able to extract different features
    def __init__(self, n_heads: int, head_size: int, seq_len: int, embed_size: int):
        # n_heads is the number of head attention
        # head_size is the hidden_size in each HeadAttention
        super().__init__()

        # FIXME:
        # head_size = embed_size // n_heads

        # Ensure embed_size is divisible by n_heads
        assert embed_size % n_heads == 0, "embed_size must be divisible by n_heads"

        # Define the head size based on the number of heads and embedding size
        self.n_heads = n_heads
        self.head_size = head_size
        self.embed_size = embed_size

        # Create multiple HeadAttention modules and store them in a ModuleList
        self.heads = nn.ModuleList(
            [HeadAttention(seq_len, embed_size, head_size) for _ in range(n_heads)]
        )

        # Define the linear layer to combine the outputs from all heads
        self.linear = nn.Linear(n_heads * head_size, embed_size)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size), make sure embed_size=n_heads x head_size
        # return: (batch_size, seq_len, embed_size)

        # FIXME:
        # Apply each HeadAttention module to the inputs
        head_outputs = [head(inputs) for head in self.heads]  # List of (batch_size, seq_len, head_size)

        # Concatenate the outputs from each head along the last dimension
        concatenated = torch.cat(head_outputs, dim=-1)  # (batch_size, seq_len, n_heads * head_size)

        # Apply the linear layer to combine the heads' outputs
        output = self.linear(concatenated)  # (batch_size, seq_len, embed_size)

        return output


class Expert(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        # FIXME:
        self.linear1 = nn.Linear(embed_size, 4 * embed_size)
        self.linear2 = nn.Linear(4 * embed_size, embed_size)
        self.activation = nn.ReLU()  # Use ReLU as the activation function

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        # -> mid: (batch_size, seq_len, 4 x embed_size)
        # -> outputs: (batch_size, seq_len, embed_size)
        # FIXME:
        # Apply the first linear transformation and activation
        mid = self.activation(self.linear1(inputs))  # (batch_size, seq_len, 4 * embed_size)
        # Apply the second linear transformation
        outputs = self.linear2(mid)  # (batch_size, seq_len, embed_size)
        return outputs


# First define the top k router module
class TopkRouter(nn.Module):
    def __init__(self, embed_size, num_experts, active_experts):
        ## FIXME:
        ## embed_size : dimension of embedding
        ## num_experts : how many Experts per layer
        ## active_experts: only active_experts out of num_experts are selected to process Embeddings per token.
        super().__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Define a linear layer to compute scores for each expert
        self.score_network = nn.Linear(embed_size, num_experts)

    def forward(self, inputs):
        ## FIXME:
        ## 完成这部分时，注意使用 Softmax() 对 router_output 做标准化。同时注意这部分所用操作的可导性。
        ## 输入值
        ## inputs is the output tensor from multihead self attention block, shape (B:batch size, T: seq_len, C: embed_size)
        ## 返回值
        ## router_output: normalized weight of Experts, 即教程中的 \alpha
        ## indices:   index of selected Experts, 即教程中的 index
        # Compute scores for each expert
        scores = self.score_network(inputs)  # (batch_size, seq_len, num_experts)

        # Get the top k scores and their indices
        topk_scores, topk_indices = torch.topk(scores, self.active_experts, dim=-1)

        # Create a mask with -inf for non-topk scores
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(-1, topk_indices, topk_scores)

        # Apply softmax to the masked scores to get normalized weights
        router_output = F.softmax(mask, dim=-1)  # (batch_size, seq_len, num_experts)

        # Create binary indices for selected experts
        indices = torch.zeros_like(scores)
        indices.scatter_(-1, topk_indices, 1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, embed_size: int, num_experts: int, active_experts: int):
        # FIXME:
        super().__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Initialize the TopkRouter
        self.router = TopkRouter(embed_size, num_experts, active_experts)

        # Initialize the Experts
        self.experts = nn.ModuleList([Expert(embed_size) for _ in range(num_experts)])

    def forward(self, inputs):
        # FIXME:
        # inputs: (batch_size, seq_len, embed_size)
        # Get the router outputs and indices of selected experts
        router_output, indices = self.router(inputs)  # (batch_size, seq_len, num_experts)
        # Initialize the final output tensor
        batch_size, seq_len, _ = inputs.shape
        final_output = torch.zeros(batch_size, seq_len, self.embed_size, device=inputs.device)

        # Iterate over the number of experts
        for i in range(self.num_experts):
            # Get the indices where the i-th expert is active
            mask = indices[..., i]  # (batch_size, seq_len)
            if mask.sum() > 0:
                # Extract the embeddings that will go through the i-th expert
                expert_inputs = inputs * mask.unsqueeze(-1)  # (batch_size, seq_len, embed_size)
                # Apply the i-th expert
                expert_output = self.experts[i](expert_inputs)  # (batch_size, seq_len, embed_size)
                # Weight the expert output with the router output and accumulate
                final_output += expert_output * router_output[..., i].unsqueeze(-1)

        return final_output


class Block(nn.Module):
    # Transformer basic block, consist of MultiHeadAttention, FeedForward and layer normalization
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        seq_len: int,
        num_experts: int,
        active_experts: int,
    ):
        super().__init__()
        # FIXME: implement block structure
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(n_heads, embed_size // n_heads, seq_len, embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.feedforward = SparseMoE(embed_size, num_experts, active_experts)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        # FIXME: forward with residual connection
        x = self.layernorm1(inputs)
        x = self.attention(x) + inputs
        y = self.layernorm2(x)
        output = self.feedforward(y) + x
        return output


class SparseMoETransformer(nn.Module):
    # Transformer decoder, consist of
    # token embedding layer and position_embedding(position_embedding 可以理解为对位置编码，感兴趣的同学可以查阅原文，这里可以看为vocab_len = seq_len的Embedding)
    # a stack of Transformer basic block
    # a layernorm and output linear layer
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_size: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        active_experts: int,
    ):
        # vocab_size is the number of word in vocabulary dict
        # seq_len is the sequence length/sentence length
        # embed_size is the embedding vector dimension
        super().__init__()
        # FIXME:
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(seq_len, embed_size)
        self.blocks = nn.ModuleList([
            Block(embed_size, n_heads, seq_len, num_experts, active_experts)
            for _ in range(n_layers)
        ])
        self.layernorm = nn.LayerNorm(embed_size)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs, labels=None):
        # labels: the (ground) true output
        # FIXME: implement the forward function of the transformer

        # inputs:(batch_size, seq_len, )
        batch_size, seq_len = inputs.shape
        # embedding:(batch_size, seq_len, embed_size)
        position_ids = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.token_embedding(inputs) + self.position_embedding(position_ids)
        # attens:(batch_size, seq_len, embed_size)
        x = embeddings
        for block in self.blocks:
            x = block(x)

        x = self.layernorm(x)
        # logits:(batch_size, seq_len, vocab_size)
        logits = self.output_layer(x)

        # compute the loss

        if labels is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            labels = labels.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        inputs = torch.tensor(tokenizer.encode(inputs)).unsqueeze(0)
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        if inputs.size(1) > self.seq_len:
            inputs = inputs[:, : self.seq_len]
        generated = inputs
        for _ in range(max_new_tokens):
            if generated.size(1) > self.seq_len:
                generated_input = generated[:, -self.seq_len :]
            else:
                generated_input = generated
            logits, _ = self.forward(generated_input)
            last_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(last_logits, dim=-1)
            next_token_ids = next_token_ids.unsqueeze(-1)
            generated = torch.cat([generated, next_token_ids], dim=1)
        return generated


from tqdm import tqdm


def train(model, dataloader, epoch, device):
    # Optimizer 会根据模型的输出和真实标签计算梯度，然后利用反向传播算法更新模型的参数。
    # 在本实验中你可以将 Optimizer 视作黑盒，只需要知道如何使用即可。
    # 找一个合适的 Optimizer。对不同的任务，模型，最适合的优化器是不一样的，你可以先尝试最常用的 Adam，如果有兴趣可以看看其他的优化器。
    # docs see: https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # FIXME:
    model.train()
    total_loss = 0

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # FIXME: implement the training process, and compute the training loss and validation loss
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, loss = model(inputs, labels=targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


def validate(model, dataloader, epoch, device):
    model.eval()
    # FIXME: 实现验证函数。与训练函数类似，但不需要计算梯度。
    total_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, labels=targets)
            total_loss += loss.item()

    print(f"Epoch {epoch} Validation Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


train_dataloader, val_dataloader = create_dataloader("input.txt", tokenizer, chunk_size=50, batch_size=512)
model = SparseMoETransformer(
    vocab_size=len(tokenizer.char2index),
    seq_len=50,
    embed_size=64,
    n_layers=3,
    n_heads=8,
    num_experts=8,
    active_experts=2,
).to(device)


# 训练模型
def run(model, train_dataloader, valid_dataloader, device, epochs=10):
    train_losses = [] # FIXME:
    valid_losses = [] # FIXME:

    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, epoch, device)
        valid_loss = validate(model, valid_dataloader, epoch, device)
        train_losses.append(train_loss) # FIXME:
        valid_losses.append(valid_loss) # FIXME:
        print(f"Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}")

    return train_losses, valid_losses


# FIXME: 用 matplotlib plot 训练过程中的 loss 变化
# run(model, dataloader, None, device, epochs=5)
train_losses, valid_losses = run(model, train_dataloader, val_dataloader, device, epochs=5)
# Plot the training and validation loss
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
print(
    tokenizer.decode(
        model.generate("I could pick my lance", max_new_tokens=100)[0]
    )
)
