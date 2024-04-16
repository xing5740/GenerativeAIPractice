# GAI 2024 Project 2.A Report

[Project Requirements](https://hackmd.io/@w-cKMxhqRAGOleYimY6D3Q/HJ-O4fOCT)
[Jupyter Notebook](https://github.com/xing5740/GenerativeAIPractice/blob/main/generativeAI_project2A.ipynb)

## Model Analysis


### Model Design

沿用sample code當中的架構，但我新增了一層RNN，也就是 embedding layer -> 3 RNN layer -> output layer

![image](https://hackmd.io/_uploads/SkgZNBteA.png)

loss function是cross entrophy, 忽略<pad>字元造成的loss，讓模型可以專注於等號後的結果，也就是我們真正想要的答案
optimizer則是常用的Adam
![image](https://hackmd.io/_uploads/r1j2MdtlR.png)
    
超參數設定如下:
![image](https://hackmd.io/_uploads/HkWh7dKeC.png)


### Process of Loss Reduction
loss在第一個epoch浮動較大，從最開始的超過1.5變成0.8，後面幾個epoch則變化較小，最後逐漸收斂至0.6
![image](https://hackmd.io/_uploads/Syz-MOYgR.png)


### Results of Validation and Testing

train, val, test的資料比例分別是0.8, 0.1, 0.1，validation和train同步進行，test則是在整個train process結束後進行

Test的方法是從test dataset中取一個batch，丟到model中進行predict，然後將predict出的y與原本的y進行比較 (只比較y當中不是<pad>的部分)，準確率大約是0.3，結果如下圖:

![image](https://hackmd.io/_uploads/HkyfX_YeR.png)



## Dataset analysis

### Characteristics

除了完整的dataset之外，我還另外生成了兩種dataset variation，我對data進行的處理如下:

- **Dataset 1: 去除乘法、保留加減法**
利用pandas.Series.str.contains對data進行filter，只留下加減法的算式
![image](https://hackmd.io/_uploads/Hkz0iHYgR.png)

- **Dataset 2: 去除Dataset 1當中極端值**
計算dataset 1的平均字串長度$\mu$與標準差$\sigma$，保留長度在$\mu - 2\sigma$到$\mu + 2\sigma$之間的字串
![image](https://hackmd.io/_uploads/rJ3cg9YlC.png)

3個資料集的訓練及測試結果如下
    
- 完整Data:
    ![image](https://hackmd.io/_uploads/S1m77khg0.png)
    ![image](https://hackmd.io/_uploads/HJb87J3x0.png)


- Dataset 1:
    ![image](https://hackmd.io/_uploads/Syz-MOYgR.png)
    ![image](https://hackmd.io/_uploads/HkyfX_YeR.png)

    
- Dataset 2:
    ![image](https://hackmd.io/_uploads/BJ7p0qYxA.png)
    ![image](https://hackmd.io/_uploads/ByWe1jFeR.png)

分成兩部分討論:
    
1. 完整Data與Variations
    
    從上面的結果可以發現，用完整Data訓練出的模型loss值較高、準確率較低。而Dataset 1測試的準確率不高，但細看model output可以發現其實與正確答案相當接近。
    
    我僅保留加減法的運算，是希望模型能夠專注於加減法運算，不要被更複雜的乘法干擾，進而達到比較好的效果。觀察上面的結果，我認為有達到我最初預期的成效。
    
2. Variations (Dataset 1 & Dataset 2)

    在dataset 2當中，我會想要以長度作為data篩選的標準，是因為我當初認為以字串的資料而言，最能夠代表的feature除了運算符號外，只剩下長度而已，所以取長度位於兩個標準差內的字串作為訓練資料。

    但從結果來看，兩者loss與準確率差異並不大，甚至dataset 2的表現比較差。我推測是因為字串長度並非模型訓練的關鍵資訊，且去除長度為極端值的資料後，資料數量下降，導致model表現變差。



## Discussion
我使用的dataset是上面的dataset 1 (加減法)，總共有三種不同超參數的model:


| Batch Size | Learning Rate (lr) |
| -------- | -------- |
| 64     | 0.001     |
| 128     | 0.001     |
| 64     | 0.002     |

3個model的訓練與測試結果如下

- batch_size=64, lr=0.001:
![image](https://hackmd.io/_uploads/Syz-MOYgR.png)
![image](https://hackmd.io/_uploads/HkyfX_YeR.png)
    
- batch_size=128, lr=0.001:
![image](https://hackmd.io/_uploads/H1QtFOKg0.png)
![image](https://hackmd.io/_uploads/HysAtdFeR.png)
    
- batch_size=64, lr=0.005:
![image](https://hackmd.io/_uploads/S1yP0OFxR.png)
![image](https://hackmd.io/_uploads/SJABCOtgC.png)


### Batch Size
    
從上面的結果可以看出，當batch size增加時，loss明顯增加，準確率明顯下降 (一題都沒答對)

猜測原因如下:
    batch size代表看了多少資料後對參數進行調整，learning rate則代表對參數進行調整的幅度
    所以當batch_size提升、learning rate不變，代表單一樣本對參數更新的影響減少，所以train的效果不明顯；或是在訓練時只看到batch內data，進而陷入局部最優解，使得輸出結果不理想



### Learning Rate

從上面的結果可以看出，當learning rate增加時，loss同樣明顯增加，準確率明顯下降 (準確率從0.3變成0.1)

猜測原因如下:
    learning rate增加表示每次對參數進行調整的幅度加大，會造成訓練過程中震盪，難以收斂至最優解，或是離最優解越來越遠，造成loss值增大，最終輸出結果不理想