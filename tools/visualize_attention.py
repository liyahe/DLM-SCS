import matplotlib as mpl
from matplotlib import pyplot as plt


def visualize_attention(attention_matrix, input_ids, mask_pos, tokenizer, layer):
    # pdata = np.rand((30, 30)) * 255
    pdata = attention_matrix.detach().numpy()
    cmap = mpl.cm.gray_r
    norm = mpl.colors.Normalize(vmin=0)
    if input_ids is not None:
        real_length = len(pdata)
        text = f'mask_pos/real_length:{mask_pos}/{real_length}, layer:{layer}\n' + str(tokenizer.decode(input_ids))
        cb = plt.text(-1, -1, s=text, fontsize=7)
    plt.imshow(pdata, cmap=cmap)
    plt.show()
