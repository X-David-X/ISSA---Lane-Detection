import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
width_video = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height_video = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

_lol_width_12 = int(width_video/12)
_lol_height_12 = int(height_video/12)

default_stanga_x_jos = 0
default_dreapta_x_sus = 450
default_dreapta_x_jos = 450

printez = True
def GreyConv(frame):
    height, width, color = frame.shape
    newFrame = np.zeros((height, width), dtype=np.uint8)
    # Grayscale = 0.299R + 0.587G + 0.114B accordin to the wavelenghts, the best
    newFrame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    newFrame = newFrame.astype(np.uint8)
    return newFrame

def TrapezCreate(frame):
    height, width, color = frame.shape
    sus_stg = np.array([int(width * 0.44), int(height * 0.75)])
    sus_drt = np.array([int(width * 0.54), int(height * 0.75)])
    jos_stg = np.array([0, height])
    jos_drt = np.array([width, height])

    trapezoid_bounds = np.array([sus_drt, sus_stg, jos_stg, jos_drt])
    bounds = np.array(trapezoid_bounds, dtype=np.int32)
    black_frame = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(black_frame, bounds, 1)

    return black_frame, trapezoid_bounds

def topdownView(frame, new_area):
    height, width = frame.shape
    frame_bounds = np.float32([np.array([width, 0]), np.array([0, 0]), np.array([0, height]), np.array([width, height])])
    magicalMatrix = cv2.getPerspectiveTransform(new_area, frame_bounds)
    newFrame = cv2.warpPerspective(frame, magicalMatrix, (width, height))

    return newFrame

def BlurredImg(frame):
    blured_frame = cv2.blur(frame, ksize=(7, 7))
    return blured_frame

def Binarizata(frame):
    newFrame = cv2.convertScaleAbs((frame > 70) * 255)
    return newFrame

def SobelFilter(frame):
    sobel_vert = np.float32([[-1, -2, -1],
                             [0, 0, 0],
                             [+1, +2, +1]])
    sobel_horiz = np.transpose(sobel_vert)
    frame = np.float32(frame)
    frame_vertical_sobel = cv2.filter2D(frame, -1, sobel_vert)
    frame_orizontal_sobel = cv2.filter2D(frame, -1, sobel_horiz)
    final_sobel = np.sqrt(frame_orizontal_sobel ** 2 + frame_vertical_sobel ** 2)
    return final_sobel

def calculCoord(frame, width):
    injumatatire = width // 2
    stanga_frame = np.copy(frame[:, :int(width//2)])
    dreapta_frame = np.copy(frame[:, int(width//2):])

    jumatate_stanga = np.argwhere(stanga_frame > 1)
    jumatate_dreapta = np.argwhere(dreapta_frame > 1)

    ys_stanga = jumatate_stanga[:, 0]
    xs_stanga = jumatate_stanga[:, 1]
    ys_dreapta = jumatate_dreapta[:, 0]
    xs_dreapta = jumatate_dreapta[:, 1] + injumatatire
    return xs_stanga, ys_stanga, xs_dreapta, ys_dreapta

def CoordLinie(frame):
    newFrame =np.argwhere(frame > 1)
    ys = newFrame[:, 0]
    xs = newFrame[:, 1]
    return xs,ys

while True:

    ret, frame = cam.read()
    if ret is False:
        break

    resized_frame = cv2.resize(frame, (320, 180))
    cv2.imshow('frame initial', resized_frame)

    gray_frame = GreyConv(resized_frame)
    cv2.imshow('gray frame', gray_frame)

    height_res, width_res = gray_frame.shape
    if(printez == True):
        print(width_res, height_res)
        printez = False
    trapezoid_frame, trapezoid_bounds = TrapezCreate(resized_frame)
    cv2.imshow('Trapezoid', trapezoid_frame * 255)

    frame_trapezoid_black = gray_frame * trapezoid_frame
    cv2.imshow( 'Drum', frame_trapezoid_black)

    trapezoid_bounds = np.float32(trapezoid_bounds)
    topDown_frame = topdownView(frame_trapezoid_black, trapezoid_bounds)
    cv2.imshow('Top Down View', topDown_frame)

    blurred_frame = BlurredImg(topDown_frame)
    cv2.imshow('Blurred', blurred_frame)

    sobel_frame = cv2.convertScaleAbs(SobelFilter(blurred_frame))
    cv2.imshow('Sobel', sobel_frame)

    frame_binarizat = Binarizata(sobel_frame)
    cv2.imshow('Frame binarizat', frame_binarizat)

    copy_frame_binarizat = frame_binarizat.copy()
    margine_stg = int(0.1 * width_res)
    margine_drt = int(0.1 * width_res)

    copy_frame_binarizat[: ,  :margine_stg] = 0
    copy_frame_binarizat[: , -margine_drt:] = 0

    left_xs, left_ys, right_xs, right_ys = calculCoord(copy_frame_binarizat, width_res)
    left_points = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_points = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    jumatate = width_res // 2

    y_sus_stg = 0  # ==  y_sus_drt =0
    y_jos_stg = height_res  # ==  y_jos_drt = heigth_res

    x_stg_sus = int((y_sus_stg - left_points[0]) / left_points[1])
    x_stg_jos = int((y_jos_stg - left_points[0]) / left_points[1])
    x_drt_sus = int((y_sus_stg - right_points[0]) / right_points[1])
    x_drt_jos = int((y_jos_stg - right_points[0]) / right_points[1])

    if 0 <= x_stg_sus < jumatate:
        default_stanga_x_sus = x_stg_sus

    if 0 <= x_stg_jos < jumatate:
        default_stanga_x_jos = x_stg_jos

    if jumatate < x_drt_sus < width_res:
        default_dreapta_x_sus = x_drt_sus

    if jumatate <= x_drt_jos < width_res:
        default_dreapta_x_jos = x_drt_jos

    # linie binarizata
    cv2.line(copy_frame_binarizat, (default_stanga_x_sus, y_sus_stg), (default_stanga_x_jos, y_jos_stg), (200, 0, 0), 5)
    cv2.line(copy_frame_binarizat, (default_dreapta_x_sus, y_sus_stg), (default_dreapta_x_jos, y_jos_stg), (100, 0, 0), 5)

    # Linii: stanga - dreapta
    frame_bounds = np.float32( [np.array([width_res, 0]), np.array([0, 0]), np.array([0, height_res]), np.array([width_res, height_res])])

    frame_temp_1 = np.zeros((height_res, width_res), dtype=np.uint8)
    cv2.line(frame_temp_1, (default_stanga_x_sus, y_sus_stg), (default_stanga_x_jos, y_jos_stg), (255, 0, 0), 5)
    magicalMatrix_left = cv2.getPerspectiveTransform(frame_bounds, trapezoid_bounds)
    frame_linie_stanga = cv2.warpPerspective(frame_temp_1, magicalMatrix_left, (width_res, height_res))

    frame_temp_2 = np.zeros((height_res, width_res), dtype=np.uint8)
    cv2.line(frame_temp_2, (default_dreapta_x_sus, y_sus_stg), (default_dreapta_x_jos, y_jos_stg), (255, 0, 0), 5)
    magicalMatrix_right = cv2.getPerspectiveTransform(frame_bounds, trapezoid_bounds)
    frame_linia_drepata = cv2.warpPerspective(frame_temp_2, magicalMatrix_right, (width_res, height_res))

    final_frame = resized_frame.copy()
    left_xs_f, left_ys_f = CoordLinie(frame_linie_stanga)
    right_xs_f, right_ys_f = CoordLinie(frame_linia_drepata)

    final_frame[left_ys_f, left_xs_f] = (50, 50, 250)
    final_frame[right_ys_f, right_xs_f] = (50, 250, 50)

    cv2.imshow('linie stg', frame_linie_stanga)
    cv2.imshow('linie drt ', frame_linia_drepata)
    cv2.imshow('Final Frame! ', final_frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
