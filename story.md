# Đề xuất định vị lại GGPKD cho ICLR — giữ nguyên method

Ngày khảo sát: 16/09/2026. Đầu vào: `Pasted markdown(1).md`, story v2 ngày 15/09/2026. Đây là phản biện và đề xuất thiết kế nghiên cứu, chưa phải kết quả thực nghiệm mới. Tôi chỉ có story được gửi, không có code, embeddings hay logs để xác minh implementation và chạy ablation.

## 1. Đề xuất chính

**Nên chuyển trọng tâm từ “partial rows are exact, graph connectivity explains recovery” sang “distilling graded neighborhood relations through sampled subgraphs”.** Điểm cần làm nổi bật là cách tổ chức tín hiệu giám sát: cho student nhìn thấy các hàng xóm cùng nhau, học sự khác biệt về mức độ tương đồng giữa chúng, và khai thác các row hợp lệ của những văn bản đã encode trong cùng pool.

Tên đề xuất:

> **From Batches to Subgraphs: Distilling Graded Neighborhoods for Text Embeddings**

Tên ngắn hơn:

> **Distilling Graded Neighborhoods for Compact Text Embeddings**

Luận điểm trung tâm:

> Để chuyển giao các phân biệt ngữ nghĩa của một embedding teacher, đưa những văn bản gần nhau vào cùng batch mới chỉ tạo ra cơ hội học. Student còn cần được giám sát bằng mức độ tương đồng tương đối bên trong từng lân cận. Chúng tôi tổ chức dữ liệu thành các subgraph của teacher, khớp phân phối hàng xóm trên mỗi row đủ điều kiện trong pool, và kết hợp với một mục tiêu trên toàn pool để giám sát các so sánh ngoài support cục bộ.

Đây là một luận điểm cần được kiểm chứng; dữ liệu trong file chưa đủ để viết “chúng tôi chứng minh thực nghiệm rằng…”. Method giữ nguyên: teacher kNN, anchor sampling, deduplicated pool, temperature hiện tại, all-eligible-row KL, pool KL và tỉ lệ loss hiện tại. Không thêm module, không thêm loss, không sửa graph hoặc sampling mặc định.

### Vì sao hướng này đáng chọn

Story cũ dựa nhiều vào một sự thật đại số đơn giản rồi kéo dài nó thành diễn giải về khả năng khôi phục hình học. Story mới đặt một câu hỏi thực nghiệm cụ thể hơn: **với cùng những embeddings đã tính, cách giám sát nào chuyển được nhiều thông tin quan hệ có ích hơn?**

Ba yếu tố có thể được phân tích riêng:

| Yếu tố | Câu hỏi | Thành phần hiện có |
|---|---|---|
| Exposure — sự hiện diện đồng thời | Các hàng xóm của cùng một tâm có xuất hiện cùng nhau không? | Pool ghép từ teacher neighborhoods |
| Resolution — độ chi tiết của target | Student học chỉ membership, hay học thứ bậc và khoảng cách similarity? | Graded conditional row KL |
| Reuse — khai thác embeddings đã tính | Những văn bản thêm vào pool có được dùng làm tâm của row riêng không? | Mọi row có ít nhất hai hàng xóm quan sát được |

Không nên quảng bá ba từ này thành ba thuật toán mới. Đây là ba trục phân tích để chỉ ra đóng góp của cấu hình hoàn chỉnh.

## 2. Những phát hiện cần sửa ngay

### 2.1. Bảng TALAS đang không khớp nguồn công khai

Table 1 trong bản PDF ACL chính thức ghi các Avg dưới đây; bản arXiv HTML tôi đọc cũng khớp các giá trị này. [TALAS, ACL 2026, Table 1](https://aclanthology.org/2026.acl-long.1509.pdf).

| Setting | TALAS trong file của bạn | TALAS trong PDF ACL |
|---|---:|---:|
| Qwen3-4B → BERT-base | 78.04 ± 0.07 | 78.83 |
| BGE-M3 → MiniLMv2-H768 | 76.64 ± 0.04 | 76.55 |
| Qwen3-0.6B → MiniLMv2-H384 | 73.46 ± 0.10 | 74.79 |

Chưa thể kết luận số trong file sai: chúng có thể là reproduction hoặc một phiên bản khác. Nhưng không thể tiếp tục ghi đơn giản “copied from TALAS paper”. Cần ghi rõ published / reproduced, phiên bản, số seeds, data split và checkpoint rule. Các số published ở bảng trên không kèm độ lệch chuẩn như bảng trong file.

Việc này ảnh hưởng trực tiếp tới C1. Câu “best in all three settings” cũng không tương đương với tiêu chí thắng ít nhất hai trên ba setting. Hãy chọn một claim nhất quán sau khi có kết quả final.

### 2.2. CoSS không phải chỉ là pointwise cosine trên neighbor batches

CoSS kết hợp feature similarity và space similarity; loss thứ hai so sánh các chiều biểu diễn trên ma trận batch. Vì vậy, “pointwise cosine + neighbor-composed batch” là đối chứng do bạn thiết kế, không phải reproduction đầy đủ của CoSS. [CoSS, §4.1–4.2](https://arxiv.org/html/2409.13939v1).

Đổi tên row đó thành **Pointwise KD + neighbor sampling**. Nếu cần kết luận so với CoSS, phải chạy adaptation đầy đủ và mô tả projection, normalization, loss weights. Một kết quả của pointwise-only không bác bỏ hiệu quả của CoSS.

### 2.3. Không thể nói các paper trước đều chỉ supervise anchors

BINGO có inter-sample loss sử dụng student embedding của positive neighbor và teacher embedding của anchor. Vì thế “neighbors chỉ là context, không được supervise” là mô tả sai. Khác biệt đáng nói hơn là **neighbor được giám sát bằng phân phối lân cận riêng của chính nó**. [BINGO, Eq. 6–9](https://arxiv.org/pdf/2107.01691).

PKT và những cách matching quan hệ trên ma trận batch cũng không cho phép claim “chúng tôi là người đầu tiên dùng mọi sample”. Hơn nữa, Jina v5 mô tả relational regularizer trên anchors, positives và negatives. [PKT](https://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html), [Jina v5, §4.2.4](https://arxiv.org/html/2602.15547v2).

### 2.4. Renormalization không biến sampled objective thành unbiased objective

Lemma trong file đúng với cùng support và cùng temperature dương ở teacher/student. Nhưng nó nói về **nghiệm loss bằng 0** trên một row, không chứng minh gradient không bias, optimizer giống full-row training, hay optimum giống nhau khi student không biểu diễn được teacher.

Không nên viết “sampled softmax is biased only because targets are not renormalized”. Nghiên cứu sampled softmax xem xét điều kiện trên phân phối lấy mẫu và gradient estimator; đây là vấn đề khác. [Blanc & Rendle, ICML 2018](https://proceedings.mlr.press/v80/blanc18a.html).

### 2.5. Connectivity không bảo đảm khôi phục các cặp ngoài graph

Reciprocal connectivity giúp đồng nhất các offset của row khi matching chính xác. Nó không xác định toàn bộ Gram matrix, không bảo đảm thứ hạng giữa neighbor và non-neighbor, và không bảo đảm held-out edge reconstruction. Mục 5 đưa ra phản ví dụ đúng với cosine và kNN.

Vì thế, không dùng “smallest k at which the graph connects” làm quy tắc suy ra từ định lý. Connectivity có thể được báo cáo như một diagnostic phụ.

### 2.6. So sánh B=64 với pool khoảng 4.7k còn confound lớn

Với số trong file, method encode khoảng **73.4 lần** số văn bản mỗi step so với baseline 64 texts. Đây là tỉ lệ số văn bản, không phải tỉ lệ FLOPs hay thời gian. Cùng optimizer steps hoặc cùng anchor epochs không có nghĩa cùng chi phí training.

Cần cả đối chứng dùng **cùng pool**, và đường quality theo tổng encoded tokens / thời gian. Báo per-step cost minh bạch là cần thiết, nhưng chưa tự giải quyết được câu hỏi “gain có phải do train nhiều hơn không?”.

## 3. Survey theo khoảng cách với method

Đây là khảo sát có mục tiêu quanh các thành phần và claim của method, cập nhật cả công trình 2025–2026 tìm thấy. Nó không phải một systematic review có bảo đảm vét hết mọi paper liên quan. Mức “gần” dưới đây là đánh giá của tôi từ các nguồn được dẫn.

| Công trình | Phần đã có trước | Ý nghĩa cho cách định vị của bạn |
|---|---|---|
| [PKT, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html) | Chuyển giao phân phối quan hệ trong feature space | KL giữa các phân phối quan hệ không mới. Cần phân biệt công thức kernel gốc với softmax-cosine adaptation. |
| [LSP, CVPR 2020](https://arxiv.org/abs/2003.10477) | Matching các phân phối local structure trên graph | Đây là antecedent trực tiếp ở cấp objective. Không nên chỉ dựa vào câu “họ whole graph, ta sampled graph” để tạo novelty. |
| [CompRess, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/975a1c8b9aee1c48d32e13ec30be7905-Abstract.html), [SEED, ICLR 2021](https://arxiv.org/abs/2101.04731) | Distill phân phối similarity trên tập reference instances | Pool KL thuộc dòng này. Các bản dùng memory queue cũng cho thấy hạn chế 64 in-batch columns không áp dụng cho toàn bộ relational KD. |
| [BINGO, ICLR 2022](https://arxiv.org/pdf/2107.01691) | Teacher-based bags và intra/inter-sample distillation | Neighborhood selection và neighbor supervision đã có. Điểm cần phân biệt là matching graded neighborhood row, thay vì bag aggregation. |
| [CoSS, 2024](https://arxiv.org/html/2409.13939v1) | Teacher kNN preprocessing, batch mở rộng, feature/space similarity | Rất gần về quy trình pool. Cần mô tả đúng cả space-similarity objective. |
| [TAS-B, SIGIR 2021](https://arxiv.org/pdf/2104.06967) | Topic-aware batches, balanced margin sampling, distillation cho retrieval | Không claim semantic batch construction là ý mới. Setting query/document và supervision khác. |
| [B3, 2025](https://arxiv.org/abs/2505.11293) | Teacher similarity graph và community-based batch construction cho contrastive learning | “Từ batch ngẫu nhiên sang graph-informed batch” không đủ mới. Phải chỉ ra thêm giá trị của target và row reuse. |
| [DistillCSE, 2023](https://arxiv.org/abs/2310.13499) | Distilled contrastive sentence learning; xử lý biến thiên logits và overfitting | Một baseline trực tiếp về sentence embeddings; không quy mọi thất bại của KD thành thiếu hàng xóm. |
| [Jasper & Stella, 2024/2025](https://arxiv.org/html/2412.19048v2) | Cosine alignment, similarity MSE và relative similarity distillation | Chuyển giao similarity grades/rankings đã có. Novelty phải nằm ở cách chọn support và dùng các partial rows trong subgraph. |
| [TALAS, ACL 2026](https://aclanthology.org/2026.acl-long.1509/) | Teacher-anchored alignment, student layer relations, ASAM | Baseline cùng miền; cần sửa provenance số liệu và đối chiếu ngân sách training. |
| [LEAF, 2025/2026](https://arxiv.org/html/2509.12539v2) | Embedding regression đơn giản, black-box teacher, teacher-aligned representations | Là đối chứng quan trọng với luận điệu rằng pointwise KD vốn yếu. Không đem leaderboard models khác dữ liệu so như controlled baseline. |
| [Jina v5, 2026](https://arxiv.org/html/2602.15547v2) | Distillation kết hợp task-specific training và relational regularization | Cần survey như công trình mới; không nhất thiết tái huấn luyện cả pipeline nếu không cùng phạm vi. |
| [Task-agnostic multi-teacher distillation, 2025](https://arxiv.org/html/2510.18680v1) | Gaussian conditional models và lập luận thông tin cho nhiều teachers | Context mở rộng về embedding KD; không phải nearest baseline về sampling. |
| [GraphSAINT, ICLR 2020](https://arxiv.org/abs/1907.04931) | Subgraph sampling và normalization | Phải cẩn thận với inclusion bias; row reweighting của bạn chỉ là adaptation, không tự động kế thừa unbiasedness. |
| [G-CRD / On Representation KD for GNNs](https://arxiv.org/abs/2111.04964) | Phân tích local/global relationship preservation | Diễn giải “local geometry + global geometry” đã đông prior art, nên không phải trục novelty tốt nhất. |

Hai nguồn lý thuyết hữu ích:

- [Hajek, Oh & Xu, NeurIPS 2014](https://arxiv.org/abs/1406.5638): partial rankings và comparison-graph spectral gap. Dùng làm bối cảnh, không dẫn như bằng chứng trực tiếp cho theorem về cosine embeddings của bạn.
- [Dong et al., NeurIPS 2023](https://arxiv.org/abs/2307.11030): RKD và clustering dưới các giả định cụ thể. Tránh lời khẳng định rộng rằng bài này là lý thuyết đầu tiên cho graph/relational KD.

Một kết nối nên thừa nhận nếu dùng KL decomposition ở mục 5 là [Decoupled KD, CVPR 2022](https://arxiv.org/abs/2203.08679). Paper đó phân tách target/non-target knowledge trong classification. Ở đây là support theo teacher neighborhoods; chỉ nên trình bày như sự tương đồng về cách phân tích, không tuyên bố chain rule mới.

**Khoảng đóng góp có thể bảo vệ:** một recipe và phân tích thực nghiệm về graded neighborhood supervision trên teacher-defined subgraphs cho compact text encoders; các partial row được tái sử dụng từ cùng pool; các yếu tố sampler, target và row coverage được kiểm soát riêng. Chưa có cơ sở để claim novelty của từng thành phần riêng lẻ hoặc “first”.

## 4. Cách kể story mới

### Mở bài bằng vấn đề đo được

Với batch ngẫu nhiên gồm B văn bản, số top-k neighbors của một anchor trong phần còn lại của batch là:

$$
X\sim\operatorname{Hypergeom}(N-1,k,B-1),\quad
\mathbb E[X]=\frac{(B-1)k}{N-1}.
$$

Với N=13,553, B=64, k=100:

| Đại lượng | Giá trị tính từ giả định lấy mẫu ngẫu nhiên |
|---|---:|
| Số neighbors kỳ vọng | 0.46488 |
| Không có neighbor nào | 62.65% |
| Đúng một neighbor | 29.47% |
| Ít nhất hai neighbors | 7.879% |

Điểm đáng kể hơn con số 0.46: **chỉ khoảng 7.9% row có đủ hai top-100 neighbors để so sánh mức độ tương đồng giữa hai hàng xóm của cùng một tâm**. Không được diễn dịch rằng 92.1% row còn lại có gradient bằng 0 hoặc không mang thông tin; chúng vẫn so sánh các văn bản khác, kể cả neighbor với non-neighbor.

Đây là động lực cho small random-batch regime, không phải định lý về toàn bộ relational distillation. Một random pool cùng cỡ 4.7k đã có khoảng 34.67 top-100 neighbors mỗi row theo kỳ vọng. Vì vậy phải kiểm tra method có lợi hơn ở cùng pool size hay không.

### Chuyển từ có hàng xóm sang học được quan hệ giữa hàng xóm

Biết rằng A, B, C cùng gần một câu không tương đương với biết B gần hơn C bao nhiêu. Method hiện tại cung cấp cả support N(j) lẫn graded distribution trên support ấy. Đây là cách giải thích tự nhiên của row loss.

Một pool cũng chứa nhiều văn bản ban đầu được đưa vào vì chúng là hàng xóm của anchors. Khi ít nhất hai hàng xóm của một văn bản như vậy cùng nằm trong pool, nó có thể tạo row supervision riêng. Method khai thác cơ hội này mà không cần thêm lượt encoder cho những văn bản đó; vẫn có thêm chi phí similarity, softmax và backward.

Pool KL bổ sung các so sánh ngoài local support. Nó cũng supervise local pairs, nên không nên mô tả hai term như hai khối thông tin hoàn toàn không chồng lấn.

### Ba claim mới

| Claim | Cách phát biểu trước khi có kết quả | Bằng chứng quyết định |
|---|---|---|
| C1: usefulness | Recipe cải thiện chất lượng student trong setting được kiểm soát, với cost minh bạch | Main results final + budget curves + baseline provenance |
| C2: graded information | Conditional teacher values mang thông tin ngoài neighbor membership | Graded / uniform / shuffled targets, giữ nguyên pool và các thành phần khác |
| C3: reuse | Giám sát row riêng của non-anchors khai thác thêm tín hiệu từ cùng embeddings | Anchors-only / all-eligible-rows, cùng pool và schedule; kiểm soát loss scale |

Exposure là động lực và một phần của C2/C3, nhưng không cần ép thành một kết luận “neighbor selection là nguồn gain lớn nhất”. Dữ liệu provisional random-columns chỉ mất khoảng 0.39 đã là lý do để không đặt cược toàn paper vào câu đó.

Các câu nên bỏ hoặc thu hẹp:

| Câu hiện tại | Thay bằng |
|---|---|
| Relational KD is starved, not biased | Small random batches provide sparse exposure to teacher-neighbor comparisons |
| Partial rows are exact constraints | Restricted rows preserve teacher log-odds on their observed support |
| Every encoded text should be a supervised row | Every eligible encoded text can contribute a neighborhood row; its benefit is tested |
| Local loss fixes local geometry | Local loss matches within-neighborhood similarity differences |
| Calibration fixes the rest | Pool KL adds broader-support similarity constraints |
| k is set by connectivity | k controls coverage, temperature and cost; choose using validation and budget |
| Global geometry preservation | Không dùng |

## 5. Phần lý thuyết nên giữ và phần nên thay

### 5.1. Lemma đúng: conditional log-odds

Với một row j, support $\Omega$, temperature chung $\tau_j>0$, đặt:

$$
p_u=\frac{\exp(s^T_{ju}/\tau_j)}{\sum_{v\in\Omega}\exp(s^T_{jv}/\tau_j)},\qquad
q_u=\frac{\exp(s^S_{ju}/\tau_j)}{\sum_{v\in\Omega}\exp(s^S_{jv}/\tau_j)}.
$$

Khi đó:

$$
\operatorname{KL}(p\|q)=0
\iff
s^S_{ju}-s^S_{jv}=s^T_{ju}-s^T_{jv}\quad\forall u,v\in\Omega.
$$

Chứng minh: KL=0 tương đương p=q; lấy log tỷ số p_u/p_v và q_u/q_v. Dạng tương đương là $s^S_{ju}-s^T_{ju}=a_j$ trên support.

**Giá trị của lemma:** giải thích chính xác thông tin được supervise. Đây là thuộc tính chuẩn của softmax; nên gọi là observation/lemma hỗ trợ method, không đặt làm phát minh toán học chính.

Tập nhiều partial supports cho cùng row có thể ghép lại nếu graph đồng xuất hiện giữa các columns liên thông: mỗi support tạo một clique trên các columns nó chứa. Khi tất cả local losses bằng 0, error của logits bằng một hằng số trên mỗi component của graph này. Nếu row đầy đủ thực sự xuất hiện, điều kiện nối các columns của row tự động được thỏa mãn.

Đó là **comparison graph trong một row**, khác với reciprocal kNN graph nối các văn bản. Việc mỗi văn bản có xác suất dương trở thành anchor giúp lập luận cho objective kỳ vọng hoặc lịch training đủ coverage, không phải bảo đảm ở mọi lịch hữu hạn có `drop_last`.

### 5.2. Phản ví dụ: cùng zero-set về mặt hình thức không bảo đảm cùng optimum khi bị giới hạn capacity

Xét teacher distribution p=(0.6,0.3,0.1) và student logits bị giới hạn là (θ,0,0), temperature 1.

- Full-row KL nhỏ nhất tại $\theta=\log 3\approx1.09861$.
- Nếu lấy hai supports {1,2} và {1,3} với xác suất bằng nhau, cả hai phía đều renormalize, kỳ vọng conditional KL nhỏ nhất khi:

$$
\sigma(\theta)=\tfrac12(\tfrac23+\tfrac67)=\tfrac{16}{21},
\quad \theta=\log(16/5)\approx1.16315.
$$

Hai optimum khác nhau. Student không thể biểu diễn p vì hai logits cuối bị buộc bằng nhau. Đó chính là kiểu vấn đề không thể bỏ qua trong model compression. Phản ví dụ này bác bỏ suy luận “renormalization nên không cần quan tâm sampling bias”; nó không bác bỏ sự hữu ích của objective hiện tại.

Phát biểu an toàn: **matching chính xác mọi observed conditional là tương thích với matching full row khi đủ coverage; các gradient và nghiệm xấp xỉ trong model class hữu hạn vẫn phụ thuộc cách lấy mẫu và trọng số.**

### 5.3. Diễn giải trực tiếp hơn: row KL là matching similarity gaps có trọng số

Đặt $\delta_u=s^S_{ju}-s^T_{ju}$. Với teacher conditional p trên support:

$$
\operatorname{KL}(p\|q)
=\log\mathbb E_{u\sim p}\exp(\delta_u/\tau_j)
-\mathbb E_{u\sim p}[\delta_u/\tau_j].
$$

Khai triển gần nghiệm matching cho:

$$
\operatorname{KL}(p\|q)
\approx\frac{1}{2\tau_j^2}\operatorname{Var}_{p}(\delta)
=\frac{1}{4\tau_j^2}\sum_{u,v\in\Omega}p_up_v(\delta_u-\delta_v)^2.
$$

Đây là suy luận đại số cho objective trong file, không phải kết quả thực nghiệm hoặc một định lý mới được lấy từ paper khác. Nó cho cách kể chuyện hữu ích:

1. **Exposure:** phải quan sát u và v cùng nhau mới có so sánh trực tiếp giữa hai similarity gaps.
2. **Resolution:** target graded quyết định cả log-odds cần khớp và trọng số của các so sánh.
3. **Reuse:** thêm eligible row tạo các so sánh quanh thêm những tâm đã được encode.

Giới hạn: xấp xỉ bậc hai chỉ dùng gần nghiệm; các constraints tương quan với nhau. Row có m columns chỉ có m−1 logit differences độc lập ở cấp score tự do, không có m(m−1)/2 constraints độc lập. KL nhỏ cũng không bảo đảm mọi gap đều nhỏ nếu xác suất target ở một số columns quá thấp.

### 5.4. KL decomposition: tại sao local conditional objective có vai trò riêng

Xét cùng một row và cùng temperature trên pool. Chia support thành A là local neighbors và C là phần còn lại. Đặt $m_T=Q^T(A)$, $m_S=Q^S(A)$. Chain rule của KL cho:

$$
\begin{aligned}
\operatorname{KL}(Q^T\|Q^S)
={}&\operatorname{KL}\big((m_T,1-m_T)\|(m_S,1-m_S)\big)\\
&+m_T\operatorname{KL}(Q^T(\cdot|A)\|Q^S(\cdot|A))\\
&+(1-m_T)\operatorname{KL}(Q^T(\cdot|C)\|Q^S(\cdot|C)).
\end{aligned}
$$

Dense pool KL đã chứa tín hiệu conditional bên trong neighborhood, nhưng trọng số của phần đó bị nhân với m_T. Một local conditional term riêng cho phép đặt trọng số trực tiếp cho các phân biệt nội bộ. **Nếu m_T nhỏ**, đây là cơ chế có thể làm local distinctions được nhấn mạnh hơn; cần đo m_T chứ không giả định nó nhỏ.

Method thật còn dùng temperature $\tau_j$ khác $\bar\tau$, đồng thời local rows và pool rows không cùng tập. Do đó không viết rằng tổng loss hiện tại “chính xác là decomposition” ở trên, hay amplifier bằng 1/m_T. Đây là công cụ phân tích một dense KL ở temperature cố định. Nếu dùng nó để giải thích kết quả, bổ sung ablation shared-temperature và đo gradient/teacher mass.

### 5.5. Phản ví dụ đúng với cosine: reciprocal connectivity không xác định non-edges

Xét N=4, k=2. Các cosine trên bốn cạnh 12,23,34,41 lần lượt là 0.20,0.21,0.22,0.23, giống nhau ở teacher/student. Hai cặp còn lại:

| Cặp | Teacher | Student |
|---|---:|---:|
| 13 | 0.02 | −0.08 |
| 24 | −0.02 | 0.08 |

Đường chéo của cả hai Gram matrices là 1. Cả hai ma trận đối xứng và positive definite: mọi tổng trị tuyệt đối ngoài đường chéo đều nhỏ hơn 1. Vì thế cả hai đều là Gram matrices hợp lệ của bốn unit vectors.

Teacher và student có cùng reciprocal top-2 graph là chu trình 1–2–3–4–1, liên thông. Mọi row KL bằng 0 vì tất cả graph-edge cosines giữ nguyên; các row temperatures đều dương. Nhưng hai non-edge similarities đã thay đổi và đảo thứ hạng.

Tôi đã kiểm tra số học: eigenvalues nhỏ nhất lần lượt khoảng 0.5691 và 0.5620; bốn row KL đều bằng 0. Đây là phản ví dụ toán học, không phải thực nghiệm trên dữ liệu của bạn.

Kết luận cần phân biệt:

- Reciprocal edges đồng nhất row offsets trong cùng component: đúng ở exact matching.
- Điều đó xác định mọi cặp trong component: sai.
- Một additive offset chung không đổi thứ hạng trong tập scores cùng chịu offset: đúng.
- Nó tự động bảo toàn ranking trên toàn corpus, gồm non-neighbors: không được suy ra.
- Các offsets và non-edges hoàn toàn tự do, độc lập: cũng quá mạnh; cosine Gram PSD, rank và shared encoder tạo thêm ràng buộc.

### 5.6. Inclusion weighting cần mô tả đúng hơn

Công thức p_j trong file đúng cho xác suất j xuất hiện trong pool khi lấy B anchors đều không hoàn lại:

$$
p_j=1-\frac{\binom{N-\operatorname{indeg}(j)-1}{B}}{\binom NB}.
$$

Nhưng row loss chỉ dùng j khi j∈R_B, tức còn cần ít nhất hai hàng xóm trong pool. Ngoài ra mẫu số |R_B| thay đổi theo batch. Hệ số row thực tế chịu ảnh hưởng của:

$$
\mathbb E\left[\frac{\mathbf1\{j\in R_B\}}{|R_B|}\,\ell_j(\Omega_j)\right].
$$

Do $\ell_j$ cũng phụ thuộc support ngẫu nhiên, chia đơn giản cho p_j không tạo unbiased estimator của uniform full-row corpus loss. Báo “inverse-inclusion reweighting ablation” sẽ chính xác hơn “GraphSAINT correction removes bias”. Giữ default hiện tại theo yêu cầu không đổi method; không tự động đưa reweighting thắng ablation thành default mới.

### 5.7. Temperature đang mang một giả định đáng nói

Với công thức hiện tại và $s^{(1)}_j>s^{(k)}_j$:

$$
\frac{P^T_j(1)}{P^T_j(k)}=
\exp\left(\frac{s^{(1)}_j-s^{(k)}_j}{\tau_j}\right)=k.
$$

Nghĩa là bạn cố định tỷ số odds giữa neighbor đầu và cuối bằng k, không cố định entropy hay perplexity. Khi sweep k, bạn đồng thời đổi graph, pool cost, target support và temperature. Bởi vậy một plateau theo k không thể được gán riêng cho connectivity.

Cần kiểm tra code đang xử lý thế nào khi top-1 và top-k có cùng similarity hoặc chênh lệch quá nhỏ. Đây là điều kiện xác định objective phải được báo cáo; tôi chưa có code để xác nhận guard hiện có.

## 6. Thí nghiệm nào thực sự làm story vững hơn?

Những thay đổi dưới đây là đối chứng và phép đo phục vụ phân tích. Chúng không thay đổi method mặc định. Không cần chạy toàn bộ trước khi biết các phép thử quyết định có ủng hộ story không.

### 6.1. Sáu phép thử ưu tiên

| Ưu tiên | Phép thử | Giữ cố định | Câu hỏi giải quyết |
|---|---|---|---|
| P0 | Rà lại nguồn baseline và protocol | Teacher/student, corpus, split, preprocessing, evaluator | Gain đang so với cái gì? |
| P1 | All eligible rows so với anchors-only | Cùng pool mỗi step, encoder calls, graph, temperature, cal loss, seeds | Non-anchor row supervision có lợi không? |
| P1 | Graded / uniform / shuffled graded targets | Cùng pool, row set, support size; chỉ đổi row target | Membership, entropy hay sự gán đúng teacher values tạo gain? |
| P1 | Random pool cùng số texts với neighborhood pool | Objective, kích thước pool, lịch đánh giá, ngân sách tokens | Lợi ích của sampler còn tồn tại khi lượng compute tương đương? |
| P1 | Chỉ pool KL, chỉ row KL, cả hai | Cùng pool và schedule | Hai term có bổ sung nhau thực nghiệm không? |
| P2 | Chất lượng theo tổng tokens và GPU-hours | Cùng hardware, cache policy, seed protocol | Lợi ích có còn khi kiểm soát training budget? |

**Shuffled graded targets** là đối chứng đặc biệt hữu ích: trên chính support quan sát được, hoán vị teacher probabilities giữa các columns. Như vậy tập neighbors, entropy và histogram probabilities giữ nguyên, nhưng gán “neighbor nào đáng tin hơn” bị phá. Nếu graded thắng shuffled, ta có bằng chứng tốt hơn cho vai trò của cấu trúc values so với chỉ graded thắng uniform.

Uniform target buộc student similarity bằng nhau trong local row ở nghiệm chính xác. Nó không tương đương mọi thuật toán binary-kNN hoặc mọi contrastive loss. Nếu chỉ uniform thua, kết luận hẹp là teacher grading hữu ích hơn equal-probability local targets trong objective này.

Trong cả uniform và shuffled arms, nếu pool KL còn giữ nguyên teacher values thì graded information vẫn tồn tại ở term ấy. Đây là thiết kế phù hợp để đo đóng góp tăng thêm của row targets, nhưng không thể kết luận rằng student hoàn toàn không còn thấy teacher grading. Có thể thêm row-only probes nếu kết quả cần được giải thích sâu hơn.

### 6.2. Ba đối chứng giúp tránh “chỉ là PKT/SEED trên batch lớn”

**Đối chứng A: random pool cùng cỡ + cùng objective.** Với M≈4.7k, lấy pool ngẫu nhiên, đặt anchors bên trong pool, giữ $\Omega_j=N(j)\cap\mathcal P$ và eligibility rule. Pool này không bảo đảm anchors có đủ top-k neighbors. Đó chính là biến cần kiểm tra: khác biệt về exposure do sampler, không phải do encode 64 so với 4.7k.

**Đối chứng B: cùng neighborhood pool, anchors-only local rows.** Đây là phép thử sạch nhất cho reuse. Tất cả văn bản vẫn được encode để làm columns. Chuyển sang all-rows thêm loss trên tâm mới, không thêm encoder pass. Tuy nhiên nó cũng đổi weighting của anchor rows; nên log loss và gradient norm, hoặc có một control kiểm tra sensitivity với loss-scale tương đương.

**Đối chứng C: dense relational KD trên cùng neighborhood pool.** Dùng mọi encoded text làm row center, so sánh với mọi text khác trong pool, với teacher temperature được mô tả rõ. Đây là baseline mạnh hơn “cal-only” hiện tại vì cal-only chỉ có B anchor rows. Dense all-rows có thể tốn M(M−1) similarities; báo cost hoặc tính theo chunks. Nếu phương án sparse local rows đạt quality tương đương với chi phí pair evaluations thấp hơn, đó là một lợi thế thực nghiệm đáng kể.

Nếu chưa đủ tài nguyên cho dense all-rows, dùng cùng tập row centers và số columns được kiểm soát để thu hẹp câu hỏi; phải nói rõ đây là baseline có ngân sách giới hạn, không phải bản dense đầy đủ.

### 6.3. Những phép đo nên có

| Mục tiêu | Phép đo | Chú ý |
|---|---|---|
| Exposure | Phân phối $|\Omega_j|$, tỉ lệ eligible rows, tỉ lệ rows có ≥2 teacher neighbors | Báo cả anchors và non-anchors |
| Teacher mass | $\sum_{u\in\Omega_j}P^T_j(u)$, và mass neighborhood trong pool softmax ở $\bar\tau$ | Đây là hai đại lượng khác nhau; không dùng chung nhãn “mass” |
| Similarity distinctions | Sai số $(s^S_{ju}-s^S_{jv})-(s^T_{ju}-s^T_{jv})$ | Dùng probes cố định, chia theo teacher gap và rank |
| Local ordering | Pairwise ordering agreement / Kendall trên neighborhood | Kèm metric theo magnitude, vì ranking không đo hết graded values |
| Neighbor membership | Teacher-neighbor recall@k của student | Không đủ một mình để chứng minh graded structure |
| Broader relations | Spearman và similarity-gap error trên các cặp outside-support | Gọi đây là fidelity/generalization probes, không phải chứng minh global recovery |
| Reuse | Eligible rows và số distinct relations được supervise trên mỗi encoded token | Không đếm repeated comparisons như độc lập |
| Cost | Encoded texts, encoded tokens, forward/backward passes, time, peak memory, pair evaluations | Teacher caching và offline graph build phải tách riêng |

Với M≈4,700 và B=64, riêng pool KL có khoảng B(M−1)=300,736 column evaluations. Bảng cost hiện tại chỉ nêu $\sum_j|\Omega_j|$ là thiếu phần này. Tổng logical evaluations trước khi tính việc tái sử dụng similarity là:

$$
\sum_{j\in R_B}|\Omega_j|+B(M-1).
$$

Nếu implementation materialize full M×M similarity rồi mới mask, actual compute/memory còn có thể khác đáng kể. Cần profile implementation, không suy runtime chỉ từ công thức sparse loss.

### 6.4. Sửa held-out edge experiment

Protocol trong file đã mask held-out pairs ở cả row loss và cal loss: đó là bước cần thiết. Nhưng cần kiểm tra thêm:

- Mask một cặp vô hướng ở cả hai chiều, tránh teacher cosine xuất hiện ở chiều ngược lại.
- Nếu withheld edges vẫn dùng để tạo pool, model vẫn nhận thông tin về membership qua sampler. Không gọi đó là “no term ever sees them” theo nghĩa không có thông tin nào lọt vào training; hãy gọi **held-out target values under a fixed graph**.
- Phân biệt thêm probe thật sự chưa dùng trong graph/sampler nếu muốn claim inductive geometry transfer. Việc tạo graph bị che edges chỉ thuộc thí nghiệm probe, không thay default method.
- Teacher scores trên held-out edges có thể đi vào $\tau_j$ hoặc $\bar\tau$. Nếu mục tiêu là strict information holdout, cần chặn đường này hoặc công khai rõ rằng đó là holdout target-only.
- Dữ liệu ngoài training corpus là phép kiểm tra generalization thuyết phục hơn reconstruction trên các văn bản đã dùng để train graph.

Không giữ prediction rằng row-only “phải ngang ours” trên same-component held-out pairs. Hãy dùng đó làm câu hỏi thực nghiệm. Nếu reciprocal graph có giant component và vài isolated nodes, cross-component probe có thể rất nhỏ và không đại diện. Báo số pairs, teacher-score distribution và uncertainty cho từng nhóm.

### 6.5. Benchmark và statistical evidence

Chín tasks đang có cho phép so sánh với protocol cũ, nhưng chưa đủ để tự động claim universal text embedding quality. Nếu story nhấn mạnh semantic neighborhoods, nên bổ sung một tập retrieval/clustering được chọn trước vì lý do nội dung, với evaluation cố định. Việc evaluate checkpoint hiện tại trên benchmark mới không thay method.

Ưu tiên ít nhất một đánh giá ngoài domain training và, nếu khả thi, một corpus distillation bổ sung lớn hơn hoặc khác domain. Đây là kiểm tra phạm vi áp dụng; không nên hứa scalability chỉ vì có thể thay exact search bằng ANN. ANN là hướng mở rộng chưa được kiểm chứng trong recipe hiện tại.

Ba seeds là mức khởi đầu. Câu “gap dưới 0.2 là noise” không phải quy tắc chung: uncertainty phụ thuộc metric, arm, covariance giữa paired seeds và checkpoint selection. Báo seed-wise paired differences, mean±sd và effect size. Dùng thêm uncertainty trên evaluation examples nếu phù hợp; nó không thay thế training seeds. Nếu claim chỉ dựa vào một gain sát noise, tăng bằng chứng cho đúng phép so sánh ấy, không mở hàng chục ablation không cần thiết.

Cần kiểm tra train/dev/test deduplication và scope của graph. Nếu graph chứa test texts, phải công khai setting transductive; không gọi nó là strict held-out inductive evaluation.

## 7. Bản story có thể thay trực tiếp cho phần đầu file hiện tại

### Working title

**From Batches to Subgraphs: Distilling Graded Neighborhoods for Text Embeddings**

### Central question

Làm thế nào để chuyển giao các phân biệt similarity của một teacher vào compact student khi mỗi bước training chỉ quan sát một phần corpus?

### Observation

Trong small random batches, một anchor hiếm khi thấy hai teacher neighbors cùng lúc. Neighborhood sampling tăng cơ hội so sánh chúng, nhưng membership không mô tả đầy đủ mức độ similarity của từng neighbor. Một pool tạo từ nhiều neighborhoods còn chứa các partial rows của nhiều văn bản ngoài anchors; những rows này tạo thêm tín hiệu có thể sử dụng từ cùng encoder outputs.

### Method

Chúng tôi xây directed teacher kNN một lần, lấy mẫu anchors và encode hợp các neighborhoods sau khi deduplicate. Mỗi encoded text có ít nhất hai teacher neighbors trong pool đóng góp một conditional row KL, dùng teacher probabilities renormalized trên support quan sát được. Một pool-wide KL trên anchors cung cấp các so sánh có support rộng hơn. Toàn bộ teacher supervision đến từ output embeddings đã cache; inference chỉ dùng student.

### What the analysis establishes

Restricted softmax matching bảo toàn teacher log-odds trên observed support khi loss bằng 0. Vì thế partial rows cung cấp ràng buộc rõ nghĩa lên similarity gaps. Kết quả này không yêu cầu gọi objective là unbiased, và không khẳng định reconstruction toàn bộ geometry.

### What the experiments must establish

1. Recipe có quality/cost tốt trong các teacher–student settings được báo cáo.
2. Graded local targets đóng góp ngoài neighbor membership và target entropy.
3. Non-anchor neighborhood rows cải thiện học ở cùng encoded pool.
4. Lợi ích của neighborhood exposure được tách khỏi việc encode nhiều văn bản hơn.

### Scope

Method bảo toàn những khía cạnh được supervise của teacher similarity, không bảo đảm mọi quan hệ teacher đều là ground-truth semantics. Cải thiện fidelity phải đi cùng cải thiện hoặc trade-off downstream được báo cáo rõ.

## 8. Nháp phần mở đầu và abstract

Các đoạn dưới đây là bản tiếng Việt để chốt lập luận trước khi viết manuscript tiếng Anh. Những câu về kết quả vẫn để placeholder; không biến số provisional thành kết luận final.

### Introduction — bốn đoạn

Các mô hình text embedding lớn biểu diễn mức độ tương đồng giữa văn bản, nhưng việc chuyển giao những phân biệt này vào một student nhỏ phụ thuộc vào cách tổ chức supervision. Một row relational loss chỉ so sánh được những văn bản xuất hiện trong support của nó. Với một corpus gồm 13,553 văn bản và batch ngẫu nhiên 64 văn bản, một anchor chỉ quan sát trung bình 0.46 trong số 100 teacher neighbors; xác suất quan sát ít nhất hai neighbors chỉ khoảng 7.9%. Trong chế độ này, các so sánh trực tiếp giữa những neighbors của cùng một tâm xuất hiện thưa thớt.

Các nghiên cứu về semantic batching và neighborhood-based distillation đã cho thấy giá trị của việc chọn những văn bản liên quan cùng xuất hiện. Tuy nhiên, co-occurrence mới quyết định quan hệ nào có thể được quan sát; nó chưa quyết định thông tin nào được truyền qua các quan hệ ấy. Neighbor membership không chỉ ra sự khác biệt về mức độ tương đồng giữa các neighbors. Đồng thời, khi nhiều neighborhoods được encode chung, pool thường chứa những partial neighborhood rows quanh các văn bản vốn chỉ được lấy vào như hàng xóm của anchors. Điều này đặt ra câu hỏi liệu khai thác các row riêng ấy có cải thiện việc chuyển giao teacher relations với cùng encoder outputs hay không.

Chúng tôi nghiên cứu một phương pháp distillation trên teacher-defined subgraphs. Mỗi training pool là hợp các teacher neighborhoods quanh một tập anchors. Sau một lần encode các văn bản duy nhất trong pool, student được giám sát bằng graded conditional distributions của mọi row đủ điều kiện. Matching một restricted row khớp các log-odds của teacher trên observed support, tạo những ràng buộc trực tiếp lên within-neighborhood similarity gaps. Một mục tiêu KL trên pool bổ sung các so sánh ngoài support cục bộ. Phương pháp không cần teacher hidden states hoặc một graph module khi inference.

Chúng tôi đánh giá phương pháp bằng [KẾT QUẢ FINAL], đồng thời tách ảnh hưởng của pool construction, teacher grading và supervised row coverage. Các thí nghiệm dùng cùng pool và các so sánh theo training budget cho thấy [CHỈ ĐIỀN NHỮNG KẾT LUẬN ĐƯỢC KIỂM CHỨNG]. Phân tích xác định rõ quan hệ nào được objective ràng buộc và những giới hạn còn lại khi chuyển giao teacher geometry sang compact students.

### Abstract — bản định hướng chưa có kết quả

Relational distillation chuyển giao sự tương đồng giữa các văn bản từ teacher sang compact student, nhưng tín hiệu học phụ thuộc vào những quan hệ cùng được quan sát và cách chúng được giám sát. Trong small random batches, các teacher neighbors của cùng một anchor ít xuất hiện đồng thời; chỉ bổ sung neighbor membership cũng chưa mô tả sự khác biệt về mức độ tương đồng giữa chúng. Chúng tôi nghiên cứu subgraph distillation với graded neighborhood targets. Mỗi bước encode hợp các teacher kNN neighborhoods và khớp conditional distribution của mọi encoded text có đủ neighbors trong pool, kết hợp với một mục tiêu trên support rộng hơn. Phân tích cho thấy partial-row matching ràng buộc teacher similarity gaps trên observed support, trong khi không đòi hỏi một tuyên bố về unbiased optimization hoặc global geometry recovery. Trên [SETTINGS], phương pháp đạt [KẾT QUẢ]. Các đối chứng ở cùng encoded pool và cùng training budget xác định [VAI TRÒ ĐÃ KIỂM CHỨNG CỦA GRADING, EXPOSURE VÀ REUSE].

## 9. Bố cục paper và hình nên làm

| Phần | Nội dung trung tâm | Bằng chứng chính |
|---|---|---|
| Introduction | Vấn đề quan sát và sử dụng graded relations | 0.46 và 7.9%, với scope nhỏ rõ ràng |
| Background / related work | Relational targets, neighborhood batches, graph sampling | Phân biệt LSP, PKT, SEED, BINGO, CoSS, B3; cập nhật text KD |
| Method | Một sơ đồ pool; hai losses; eligibility | Pseudocode đúng implementation |
| Analysis | Conditional log-odds và gap interpretation | Lemma ngắn; assumptions và limitations |
| Main results | Quality, teacher/student settings, budget | Published/reproduced tách biệt |
| Mechanism | Exposure, grading, reuse | Same-pool và matched-budget controls |
| Limitations | Corpus scale, compute, weighting, weak tasks | Không giấu WiC hoặc regression bất lợi |

**Figure 1:** dùng cùng một tập văn bản minh họa ba trạng thái: random batch thiếu co-occurring neighbors; neighborhood pool nhưng chỉ anchors có row targets; cùng pool với các eligible non-anchor rows. Dùng độ đậm hoặc nhãn probability để chỉ graded values. Caption không viết gain khi chưa có kết quả.

**Figure 2:** đường performance theo encoded tokens hoặc GPU-hours, đánh dấu main setting. Không dùng optimizer step làm trục duy nhất.

**Figure 3:** cùng neighbor recall nhưng khác similarity-gap fidelity nếu thực nghiệm thực sự cho thấy hiện tượng đó. Không dựng hình giả theo kết luận muốn có. Nếu không có sự tách biệt này, dùng trực tiếp graded/uniform/shuffled ablation.

Theory về reciprocal components và k-connectivity đưa vào appendix hoặc bỏ nếu không tạo insight thực nghiệm bổ sung. Không cần một figure connectivity chỉ để bảo vệ story cũ.

## 10. Quy tắc quyết định sau các runs

| Kết quả | Điều nên viết |
|---|---|
| All-rows thắng anchors-only rõ ràng ở cùng pool | Reuse là contribution thực nghiệm chính |
| All-rows ngang anchors-only | Giữ method, bỏ claim reuse tạo gain; thừa nhận chưa có bằng chứng cho lợi ích của thành phần này |
| Graded thắng uniform và shuffled | Có bằng chứng cho việc gán đúng graded values |
| Graded thắng uniform nhưng ngang shuffled | Chưa chứng minh teacher ordering; entropy/regularization là giải thích cạnh tranh |
| Neighborhood pool thắng random pool cùng cỡ | Có bằng chứng cho structured exposure |
| Random pool cùng cỡ ngang method | Không đặt neighborhood sampler ở trung tâm causal claim |
| Dense all-rows KD ngang method nhưng tốn pair compute hơn | Có thể định vị sparse supervision nếu đo được lợi ích cost thực |
| Cal-only ngang full method | Không claim local term giải thích gain; giữ method theo yêu cầu nhưng hạ claim tương ứng |
| Thắng theo steps nhưng không theo tokens/time | Báo trade-off; không claim training efficiency |
| Chỉ thắng STS | Thu hẹp claim sang similarity-sensitive tasks nếu các kiểm tra khác ủng hộ |
| Fidelity tăng nhưng downstream không tăng | Báo giới hạn của teacher imitation; không đồng nhất fidelity với task quality |

**Thứ tự chạy nên đổi so với file hiện tại:** kiểm tra provenance/protocol → all-rows vs anchors-only và graded vs shuffled → random pool cùng cỡ + cal/row decomposition → main settings final → cost curves → k/weight robustness. Không dành lượt chạy đầu tiên cho connectivity sweep khi chính diễn giải lý thuyết của nó chưa đứng vững.

Giữ nguyên method không đồng nghĩa giữ nguyên mọi lời giải thích về method. Giá trị của vòng nghiên cứu này là tìm ra claim nào thực sự được dữ liệu bảo vệ, thay vì làm cho mọi kết quả đều trông như ủng hộ một câu chuyện đã chốt.

## 11. Đánh giá thẳng về độ sẵn sàng cho ICLR

Hiện tại, file cung cấp một recipe có thể kiểm tra được và một số tín hiệu sơ bộ, nhưng **chưa có final results cho method đang định nghĩa**. Vì vậy chưa thể kết luận rằng đổi story là đủ để tạo một submission mạnh.

Điểm yếu dễ bị reviewer đánh vào nhất không phải tên paper, mà là: novelty của thành phần riêng lẻ thấp; baseline được đặt tên chưa đúng; large-pool compute chưa kiểm soát; và theory hiện tại có suy luận quá mức. Những điểm này đều có thể được xử lý mà không đổi method: thu hẹp claim, sửa attribution, bổ sung controlled comparisons, và thay connectivity narrative bằng phân tích conditional similarity gaps.

Tôi ưu tiên **From Batches to Subgraphs** làm khung manuscript, với **graded neighborhood relations** là đối tượng được chuyển giao và **same-pool row reuse** là claim thực nghiệm cần kiểm chứng đầu tiên. Nếu hai phép thử grading và reuse đều cho kết quả rõ, bạn sẽ có một paper xoay quanh một nguyên lý học có thể kiểm chứng. Nếu chúng không cho kết quả rõ, cách định vị trung thực còn lại là một recipe thực dụng với trade-off được đo tốt; không nên dùng lemma để bù cho bằng chứng thực nghiệm thiếu.
