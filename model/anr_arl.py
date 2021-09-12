import torch

from model.anr_config import AnrConfig


class AmrArl(torch.nn.Module):
    """
    Aspect-based Representation Learning (ARL)
    """

    def __init__(self, config: AnrConfig):
        super().__init__()
        self.config = config

        # Aspect Embeddings
        self.aspEmbed = torch.nn.Embedding(self.config.num_aspects, self.config.ctx_win_size * self.config.h1)
        self.aspEmbed.weight.requires_grad = True

        # Aspect-Specific Projection Matrices
        self.aspProj = torch.nn.Parameter(torch.Tensor(self.config.num_aspects, self.config.word_dim, self.config.h1), requires_grad=True)

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.aspEmbed.weight.data.uniform_(-0.01, 0.01)
        self.aspProj.data.uniform_(-0.01, 0.01)

    def forward(self, review_emb):
        """
        :param review_emb:  (batch size, review length, word dim)
        :return:            (batch size, num aspects, h1)
        """

        # Loop over all aspects
        lst_batch_aspAttn = []
        lst_batch_aspRep = []
        for a in range(self.config.num_aspects):
            # Aspect-Specific Projection of Input Word Embeddings: (bsz x max_doc_len x h1)
            batch_aspProjDoc = torch.matmul(review_emb, self.aspProj[a])

            # Aspect Embedding: (bsz x h1 x 1) after transposing!
            bsz = review_emb.size()[0]
            batch_aspEmbed = self.aspEmbed(torch.LongTensor(bsz, 1).fill_(a).to(self.config.device))
            batch_aspEmbed = torch.transpose(batch_aspEmbed, 1, 2)

            # Context-based Word Importance
            # Calculate Attention based on the word itself, and the (self.config.ctx_win_size - 1) / 2 word(s) before & after it
            # Pad the document
            pad_size = int((self.config.ctx_win_size - 1) / 2)
            batch_aspProjDoc_padded = torch.nn.functional.pad(batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0)

            # Use "sliding window" using stride of 1 (word at a time) to generate word chunks of ctx_win_size
            # (bsz x max_doc_len x h1) -> (bsz x max_doc_len x (ctx_win_size x h1))
            batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(1, self.config.ctx_win_size, 1)
            batch_aspProjDoc_padded = torch.transpose(batch_aspProjDoc_padded, 2, 3)
            batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(-1, self.config.review_length, self.config.ctx_win_size * self.config.h1)

            # Calculate Attention: Inner Product & Softmax
            # (bsz x max_doc_len x (ctx_win_size x h1)) x (bsz x (ctx_win_size x h1) x 1) -> (bsz x max_doc_len x 1)
            batch_aspAttn = torch.matmul(batch_aspProjDoc_padded, batch_aspEmbed)
            batch_aspAttn = torch.softmax(batch_aspAttn, dim=1)

            # Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
            # (bsz x max_doc_len x 1) and (bsz x max_doc_len x h1) -> (bsz x h1)
            batch_aspRep = batch_aspProjDoc * batch_aspAttn.expand_as(batch_aspProjDoc)
            batch_aspRep = torch.sum(batch_aspRep, dim=1)

            # Store the results (Attention & Representation) for this aspect
            lst_batch_aspAttn.append(torch.transpose(batch_aspAttn, 1, 2))
            lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))

        # Reshape the Attentions & Representations
        # batch_aspAttn:	(bsz x num_aspects x max_doc_len)
        # batch_aspRep:		(bsz x num_aspects x h1)
        batch_aspAttn = torch.cat(lst_batch_aspAttn, dim=1)
        batch_aspRep = torch.cat(lst_batch_aspRep, dim=1)

        # Returns the aspect-level attention over document words, and the aspect-based representations
        return batch_aspAttn, batch_aspRep
