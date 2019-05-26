def get_color_in_hls(rgb_color):
    r = rgb_color[0]
    g = rgb_color[1]
    b = rgb_color[2]

    v_max = max(rgb_color)
    v_min = min(rgb_color)

    if v_max == r:
        h = 30 * (g - b) / (v_max - v_min)
    elif v_max == g:
        h = 60 + 30 * (b - r) / (v_max - v_min)
    else:
        h = 120 + 30 * (r - g) / (v_max - v_min)

    l = (v_max + v_min) / 2

    if l < 0.5:
        s = (v_max - v_min) / (v_max + v_min)
    else:
        s = (v_max - v_min) / (2 - v_max + v_min)

    return [h, l, s]

#
# def hls_select(img, channel=0, thresh=(0, 255)):
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     ch = hls[:,:,channel]
#     binary_output = np.zeros_like(ch)
#     binary_output[(ch > thresh[0]) & (ch <= thresh[1])] = 1
#     return binary_output#
#
# def pipeline(input_image):
#     print("hi")

print(get_color_in_hls([230,254,244]))