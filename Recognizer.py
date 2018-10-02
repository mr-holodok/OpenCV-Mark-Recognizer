import numpy as np
import cv2


class Recognizer():
    """class with functions for recognizing answers in square boxes on blank"""

    def find_approx_page_cntr(self, image_path):
        # load the image, convert it to gray, blur it
        image = cv2.imread(image_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        # perform Otsu thresholding to detect countour
        ret, th2 = cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # find contours in the image
        img, cnts, hierarchy = cv2.findContours(th2.copy(), cv2.RETR_EXTERNAL,\
	                            cv2.CHAIN_APPROX_SIMPLE)
        # ensure that at least one contour was found
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
            # sort the contours according to their size in descending order
            for c in cnts:
                # approximate the contour
                epsilon = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * epsilon, True)
                if len(approx) == 4:
                # if our approximated contour has four points,
		        # then we can assume we have found the paper
                    #for p in approx[:,0]:    # drawing points for testing
                    #    image = cv2.circle(image,(p[0], p[1]), 10, (255,0,0), -1)
                    return approx[:,0]
        # else:        
        #raise Exception


    def sort_points(self, pts):
        # sort the points based on x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points 
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # perform the same for right-most coordinates
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        return np.array([tl, tr, br, bl], dtype="float32")


    def perspective_transformation(self, image, pts):
        # obtain a consistent order of the points and unpack them individually 
        rect = self.sort_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped


    def find_quest_cntrs(self, page_image):
        # method that finds and returns contours that represent a question squares
        self.page_image = page_image
        gray_img = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        self.thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV, 11, 2)
        cntrs, hierarchy = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:3]
        questionCntrs = [] 
        # loop over the contours
        for i, c in enumerate(cntrs):
            # it cann't be a square
            if len(c) < 4: continue  
            # prefers outer countours, (that have not a parent countour) because
            # open cv recognizes shape as 2 countors
            if hierarchy[0][i][-1] != -1: continue 
            # perimeter
            epsilon = cv2.arcLength(c, True)  
            # get approximate contour
            approx = cv2.approxPolyDP(c, 0.02 * epsilon, True)
            # it isn't a square
            if len(approx) != 4: continue    
	        # compute the bounding box of the contour, then use the
	        # bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # additional check to verify that countour really answer square
            if self.check_if_square(approx, w) and w > 100 and h > 100 and ar >= 0.9 and ar <= 1.1 :
                questionCntrs.append(approx)
                #for point in approx[:,0]:   # draw countour points for testing
                #    page_image = cv2.circle(page_image,(point[0], point[1]), 15, (255,0,0), 4)
        
        return questionCntrs 


    def check_if_square(self, cntr, width):
        # method that calculate width of every line of countour and compares it to parameter <width> 
        for i in range(0, len(cntr)):
            sq = (cntr[i][0][0] - cntr[(i+1)%len(cntr)][0][0])**2 + (cntr[i][0][1] - cntr[(i+1)%len(cntr)][0][1])**2
            sq = sq ** 0.5
            if sq < 0.9 * width or sq > 1.1 * width: return False
        return True


    def sort_contours_vrtl(self, cnts):
        # method that constructs the list of contours and sort them from top to bottom
        # create a list with bounding boxes for every contour
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        # sort the contours by y-axis of bounding box
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),\
                                            key=lambda b: b[1][1], reverse=False))
        return cnts


    def sort_contours_hrzl(self, cnts):
        # method that construct the list of bounding boxes and sort them from left to right
        # create a list with bounding boxes for every contour
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        # sort the contours by x-axis of bounding box
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),\
                                            key=lambda b: b[1][0], reverse=False))
        return cnts


    def find_answers(self, cntrs):
        cntrs = self.sort_contours_vrtl(cntrs)
        answers = []
        for (i,j) in enumerate(np.arange(0, len(cntrs), 5), 1):
            cntrs_in_row = self.sort_contours_hrzl(cntrs[j:j+5])
            for num, c in enumerate(cntrs_in_row, 1):
		        # construct a mask that reveals border of a question box
                border_mask = np.zeros(self.thresh.shape, dtype="uint8")
                cv2.drawContours(border_mask, [c], -1, 255, 40)
                # apply the mask to the thresholded image, then
		        # count the number of non-zero pixels in the square area
                border_mask = cv2.bitwise_and(self.thresh, self.thresh, mask=border_mask)
                border = cv2.countNonZero(border_mask)
                
                # construct a mask that reveals only the current square box
                mask = np.zeros(self.thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
		        # apply the mask to the thresholded image, then
		        # count the number of non-zero pixels in the square area
                mask = cv2.bitwise_and(self.thresh, self.thresh, mask=mask)
                filled = cv2.countNonZero(mask)
                
                # if the <filled> has a larger number of total non-zero pixels, than in the <border>,
                # then we are find non-empty box
                if border < filled :
                    answers.append((i, num))
                    for k in range(0,4):   # draw countour points for testing
                        self.page_image = cv2.line(self.page_image, (c[k][0][0], c[k][0][1]), (c[(k+1)%4][0][0], c[(k+1)%4][0][1]), (255,0,0), 4)
        return answers, self.page_image


    def recognize(self, image_path):
        cntr = self.find_approx_page_cntr(image_path)
        img = self.perspective_transformation(cv2.imread(image_path), cntr)
        cntrs = self.find_quest_cntrs(img)
        answ, img = self.find_answers(cntrs)
        new_path = image_path[:image_path.rfind('.')] + '_res' + image_path[image_path.rfind('.'):]
        cv2.imwrite(new_path, img)

        return answ, new_path


class ImageRotator():
    """class with function to rotate image on 90 degrees to right and return path to that image"""

    @staticmethod
    def rotate(image_path):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        center = (w/2, h/2) 
        # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        M = cv2.getRotationMatrix2D(center, -90, 1.)
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(M[0,0])
        abs_sin = abs(M[0,1])
        # find the new width and height bounds
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        M[0, 2] += bound_w/2 - center[0]
        M[1, 2] += bound_h/2 - center[1]
        # rotate image with the new bounds and translated rotation matrix
        img = cv2.warpAffine(img, M, (bound_w, bound_h))
        if image_path[image_path.rfind('/')+1:].find('_rotated') == -1:
            new_path = image_path[:image_path.rfind('.')] + '_rotated' + image_path[image_path.rfind('.'):]
        else:
            new_path = image_path
        cv2.imwrite(new_path, img)
        return new_path



if __name__ == '__main__':
    
    recognizer = Recognizer()
    answ, img = recognizer.recognize("../photo tests/t6.jpg")
    for row in answ:
        print(row)
    img = cv2.resize(img, (500,700))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()