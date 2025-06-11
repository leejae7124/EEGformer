# This is the script of EEG-Deformer
# This is the network script
# MWL Task
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module): #Transformer의 MLP 블록 역할.
    # nn.LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), #입력을 정규화하여 학습 안정성을 높인다.
            nn.Linear(dim, hidden_dim), #입력 차원을 확장한다. 정보 표현력을 넓혀줌!
            nn.GELU(), #비선형 활성화 함수로 ReLU보다 부드럽고, 트랜스포머에서 자주 쓰인다.
            nn.Dropout(dropout), #과적합 방지
            nn.Linear(hidden_dim, dim), #다시 원래 차원으로 줄여준다.(복원)
            nn.Dropout(dropout) #또 한 번 regularization
        ) #즉, attention이 뽑아낸 feature를 "조금 더 복잡한 방식으로 가공해서" 모델이 더 풍부한 표현을 학습하도록 도와주는 블록!
        #왜 필요한가? Attention은 주로 "위치 간 관계(feature 간 상호작용)"를 처리하지만, FeedForward는 각 위치(토큰, 채널 등)의 정보 자체를 강화한다.
        #이 둘을 결합하면 공간 + 의미 정보 모두 잘 처리할 수 있게 되는 것이다.

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module): #Multi-head self-attention을 구현한 모듈.
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class IPValueAttention(nn.Module):
    def __init__(self, d_embed=64, heads=4): #d_latent : 어텐션이 작동하는 공통 latent 공간의 차원, heads: 멀티 헤드 어텐션의 헤드 수
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=False) #선형 변환
        self.v_proj = nn.Linear(d_embed,  d_embed, bias=False) # 선형 변환
        self.register_buffer("fixed_key", torch.ones(1, 1, d_embed)) #모든 쿼리 위치에 대해 동일한 키 사용. attention score는 오직 쿼리(HCT) 특성에 따라 결정된다.
        self.attn = nn.MultiheadAttention(d_embed, heads, batch_first=True)
        self.out  = nn.Identity()

    def forward(self, hct_tok, ip_vec):
        B, T, _ = hct_tok.shape 
        Q = self.q_proj(hct_tok)                 # (B,T,d), hct 출력을 쿼리로 변환
        K = self.fixed_key.expand(B, T, -1)      # (B,T,d) 키 고정
        V = self.v_proj(ip_vec).unsqueeze(1).expand(-1, T, -1)  # (B,T,d)     # (B,1,d) , ip vector를 attention value로 사용
        attn_out, _ = self.attn(Q, K, V)         # (B,T,d), hct 위치마다 ip에서 필요한 정보 추출
        return hct_tok + self.out(attn_out)      # 원래 hct 출력에 residual 보강

class IPQueryAttention(nn.Module):
    """
    • Q  ← IP  (B, d_ip)           ──>  (B, 1, d_latent)
    • K  ← 고정(1)                 ──>  (B, T, d_latent)
    • V  ← HCT tokens (B, T, d_hct) ──>  (B, T, d_latent)

    출력: IP 기반으로 가중합된 HCT 요약 (B, d_latent)
    """
    def __init__(self, d_hct = 64, d_ip = 64, d_latent = 64, heads=4): #d_latent : 어텐션이 작동하는 공통 latent 공간의 차원, heads: 멀티 헤드 어텐션의 헤드 수
        super().__init__()
        self.q_proj = nn.Linear(d_ip, d_latent, bias=False) #선형 변환
        # self.k_proj = nn.Linear(d_embed,  d_embed, bias=False) # 선형 변환
        self.v_proj = nn.Linear(d_hct, d_latent, bias=False) # 선형 변환
        self.register_buffer("fixed_key", torch.ones(1, 1, d_latent)) #고정 Key(=1). 길이는 hct token 길이 T에 맞춰 runtime에서 expand

        self.attn = nn.MultiheadAttention(d_latent, heads, batch_first=True)
        self.out_proj = nn.Linear(d_latent, d_hct, bias=False)

    def forward(self, hct_tok, ip_vec):
        """
        hct_tok : (B, T, d_hct)  - 각 HCT 블록의 토큰
        ip_vec  : (B, d_ip)      - IP 파워 벡터
        return  : (B, d_latent)  - IP 를 Query 로 한 HCT 요약
        """
        B, T, _ = hct_tok.shape
        # 1) Q : IP 벡터 → (B, 1, d_latent)
        Q = self.q_proj(ip_vec).unsqueeze(1)       # (B, 1, d)
        # 2) K : 고정 1 → (B, T, d_latent)
        K = self.fixed_key.expand(B, T, -1)
         # 3) V : HCT 토큰 → (B, T, d_latent)
        V = self.v_proj(hct_tok)

        # 4) Cross-Attention
        attn_out, _ = self.attn(Q, K, V)          # (B, 1, d_latent)      # (B, 1, d_hct)
        attn_out = attn_out.expand(-1, T, -1)     # (B, T, d_hct)  ← 토큰 길이에 맞춰 broadcast

        # 5) Residual-add (옵션)
        return hct_tok + attn_out


class Transformer(nn.Module): #Deformer의 핵심 구조
    # CNN과 Transformer를 결합한 Hybrid Layer를 여러 층 쌓은 구조.
    # nn.Sequential(): Pytorch에서 여러 레이어를 순차적으로 묶어주는 컨테이너.
    def cnn_block(self, in_chan, kernel_size, dp): # fine-grained를 위해 병렬적으로 구성된 cnn 블록
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.): #Atteitnion + CNN 통합 구조를 "정의!!"
        #파라미터 설명
        # dim: 입력 feature의 차원 (시간 feature 길이 등), depth: Transformer 블록 몇 층 쌓을지, heads: Multi=head attention의 head 수
        # dim_head: 각 head당 차원, mlp_dim: FeedForward 레이어의 은닉 크기, in_chan: CNN의 입력 채널 수(eeg 채널 수?)
        # fine_grained_kernel: CNN에 사용할 커널 크기, dropout: 드롭아웃 비율
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth): #depth는 HCT 블록 갯수랑 똑같음!! 정의함.
            dim = int(dim * 0.5) #층이 깊어질수록 차원을 절반씩 줄인다. Transformer가 점점 압축된 feature를 학습
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ])) #하나의 HCT 블록 같음. 포함된 모듈: Attention, FeedForward, cnn_block
            #다음과 같은 구조가 된다.
            #             [
            # [attn1, ff1, cnn1],
            # [attn2, ff2, cnn2],
            # ...
            # ]

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) #입력 feature를 다운샘플링. coarse feature 추출에 사용?
        
        # self.hct_output_dims = [num_kernel * int(init_dim * (0.5 ** (i + 1))) for i in range(depth)]
        # print("Calculated hct_output_dims:", self.hct_output_dims)  # ← 여기서 출력

        # self.projection1 = nn.Sequential(#방법 (1): hct 각 블록의 output 차원을 동일하게 맞춤. Attention X.
        #     nn.Linear(6144, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        # self.projection2 = nn.Sequential(
        #     nn.Linear(3072, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        # self.projection3 = nn.Sequential(
        #     nn.Linear(1536, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        # self.projection4 = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        self.projection1 = nn.Sequential(#방법 (2): hct 각 블록의 output 차원을 비율에 맞게 맞춤. Attention X. (save 3)
            nn.Linear(32000, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.projection2 = nn.Sequential(
            nn.Linear(16000, 192),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.projection3 = nn.Sequential(
            nn.Linear(8000, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.projection4 = nn.Sequential(
            nn.Linear(3968, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.ip_attns = nn.ModuleList([
            IPValueAttention(d_embed=64, heads=4),
            IPValueAttention(d_embed=64, heads=4),
            IPValueAttention(d_embed=64, heads=4),
            IPValueAttention(d_embed=64, heads=4)
            # IPQueryAttention(),
            # IPQueryAttention(),
            # IPQueryAttention(),
            # IPQueryAttention()
        ])


        # self.projection1 = nn.Sequential(#방법 (3): hct 각 블록의 output 차원을 ip 차원과 동일하게 맞춘다(같은 latent space에 위치하도록). Attention X. (save 3)
        #     nn.Linear(6144, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256)
        # )
        # self.projection2 = nn.Sequential(
        #     nn.Linear(3072, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256)
        # )
        # self.projection3 = nn.Sequential(
        #     nn.Linear(1536, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256)
        # )
        # self.projection4 = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256)
        # )
        # self.IP_projection = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )


    def forward(self, x): #x의 shape는 (batch_size, in_chan, time_steps) 형태의 시계열 입력 -> EEG 데이터의 CNN 전처리 이후의 feature
        #forward는 데이터를 실행함. 앞에서 정의한 블록을 "하나씩 꺼내서 입력 데이터(x)"에 적용하는 부분. 즉, 실제 forward pass(순전파) 때 실행되는 "데이터 흐름 처리 로직"
        dense_feature = [] #각 블록에서 추출한 요약 정보(x_info)를 저장할 리스트(나중에 concatenation용)
        for i, (attn, ff, cnn) in enumerate(self.layers): #정의한 블록수만큼? 반복
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan), IP 모듈?

            
            # dense_feature.append(x_info)
            x = ff(x_cg) + x_fg #hct block i의 출력
            # if i == 0:
            #     x_proj = self.projection1(x.view(x.size(0), -1))
            # elif i == 1:
            #     x_proj = self.projection2(x.view(x.size(0), -1))
            # elif i == 2:
            #     x_proj = self.projection3(x.view(x.size(0), -1))
            # elif i == 3:
            #     x_proj = self.projection4(x.view(x.size(0), -1))
            attn_out = self.ip_attns[i](
                hct_tok = x.permute(0, 2, 1),  # (B, T, C)
                ip_vec=x_info                  # (B, 64)
            ).squeeze(1)
            if i == 0:
                attn_out = self.projection1(attn_out.flatten(1))
            elif i == 1:
                attn_out = self.projection2(attn_out.flatten(1))
            elif i == 2:
                attn_out = self.projection3(attn_out.flatten(1))
            elif i == 3:
                attn_out = self.projection4(attn_out.flatten(1))
            combined = torch.cat([attn_out, x_info], dim=-1)
            dense_feature.append(combined)
            
        # x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth, 각 블록에서 생성한 x_info들을 이어붙임. ip 모듈의 결과를 concat!
        # x = x.view(x.size(0), -1)   # b, in_chan*d_hidden_last_layer, 전체 시퀀스를 평탄화
        emd = torch.cat(dense_feature, dim=-1)  # b, in_chan*(depth + d_hidden_last_layer), 그냥 output과 ip 모듈 output을 연결하여 최종 벡터 완성
        return emd

    def get_info(self, x): #신호 요약 정보 추출. CNN 출력을 기반으로 각 채널별 에너지 수준 요약을 구한다. IP 모듈?
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1)) #제곱(pow 2)해서 파워 구하고, 길이 방향으로 평균(평균 파워) 구하고, 로그 취한다.
        return x

    def get_padding_1D(self, kernel): #1d convolution에서 출력 길이를 입력과 같게 유지하기 위해, padding을 해줌!
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d): #Cov2d에 L2 norm 제약을 추가해서, 가중치 폭주를 방지하고 일반화 성능을 높이기 위한 목적.
    #L2 norm이란? 딥러닝에서 "가중치가 너무 커지지 않도록 제어"하는 정규화 기법 중 하나. 특히 모델이 과적합되는 것을 방지하거나 안정적인 학습을 유도할 때 유용하다.
    #Conv2dWithConstraint(nn.Conv2d): 2d cnn을 상속하면서 가중치(norm)를 제한하는 기능을 추가한 커스텀 합성곱 계층.
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm #max_norm: 가중치의 L2 norm의 최대값.
        self.doWeightNorm = doWeightNorm #doWeightNorm: 가중치 정규화를 적용할지 말지 여부
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs) #nn.Conv2d의 기본 초기화(커널 크기, 채널 수 등 설정)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            ) #remorm(): 특정 차원(dim=0)을 기준으로 weight 텐서의 L2 norm을 max_norm 이하로 제한. 이후 원래의 Conv2d.corward() 수행.
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    #전체 클래스 구조 요약: (1) CNN 인코더: eeg 데이터를 spatial-temporal feature로 추출. (2) Transformer 모듈: attention + CNN으로 feature 간 상호작용 학습. (3) MLP 분류기: 최종 예측값 생성.
    def cnn_block(self, out_chan, kernel_size, num_chan): #eeg 입력은 (b, 1, 채널수, 시간) 형태
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2), #시간 방향 필터링
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2), # 모든 채널을 한 번에 처리하는 공간 필터링
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.): #전체 구조 초기화
        super().__init__()

        self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan) #cnn 인코더로 전처리

        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)') #cnn 출력을 [B, 채널, 1, 시간] -> [B, 채널, 시간] -> [B, 채널, feature]러 바꿔줌.

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim)) #위치 정보 삽입

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        ) #앞서 정의 한 transformer 모듈: HCT 구조

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth) #transformer 마지막 레이어에서 나온 feature의 크기 + 각 hct 블록에서 나온 dense summary(x_info)들을 concat한 것.

        self.mlp_head = nn.Sequential(
            nn.Linear(896, num_classes)
        ) #모델의 classifier head. 

    def forward(self, eeg):
        # eeg: (b, chan, time)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x) #flatten 후, transformer 준비

        b, n, _ = x.shape
        x += self.pos_embedding #위치 정보 추가
        x = self.transformer(x) #hct 처리
        return self.mlp_head(x) #분류 결과

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer): #transformer 블록을 통과할 때마다 feature의 크기를 절반씩 줄이는 걸 반영해서, 각 층의 hidden size를 리스트로 반환한다.
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 32, 1000)) #입력 텐서. (batch size, 채널 수, 한 채널당 시간 축 길이 몇 개의 시계열 데이터 포인트를 갖고 있는가.)
    emt = Deformer(num_chan=32, num_time=1000, temporal_kernel=11, num_kernel=64,
                 num_classes=2, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.5) #deformer 모델 객체 생성. 
    print(emt)
    print(count_parameters(emt)) #학습 가능한 파라미터 수 출력.

    out = emt(data)
