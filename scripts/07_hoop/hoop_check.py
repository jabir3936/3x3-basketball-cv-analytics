import cv2, math

H_inv = np.linalg.inv(H)

def court_to_pixel(c):
    v = np.array([c[0], c[1], 1.0], dtype=np.float32)
    t = np.dot(H_inv, v)
    return (int(t[0] / t[2]), int(t[1] / t[2]))

img = cv2.imread("media/checks/hoop_frame_grid.jpg")

hoop = [ux, uy]   # from Option A or B
cv2.circle(img, court_to_pixel(hoop), 6, (0, 0, 255), -1)

for deg in range(0, 360, 5):
    a = math.radians(deg)
    pt = court_to_pixel([hoop[0] + 6.75 * math.cos(a), hoop[1] + 6.75 * math.sin(a)])
    if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
        cv2.circle(img, pt, 2, (0, 255, 0), -1)

cv2.imwrite("media/checks/hoop_check.jpg", img)