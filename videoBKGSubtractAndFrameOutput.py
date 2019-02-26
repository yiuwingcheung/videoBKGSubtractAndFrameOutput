import cv2, os
from datetime import datetime as dt
import numpy as np

def createDir(path, isFolder):
	if isFolder and path[-1] != '/':
		path += '/'
	slashes = [p for p, c in enumerate(path) if c == '/']
	for slash in slashes:
		if not os.path.exists(path[0:slash+1]):
			os.mkdir(path[0:slash+1])
	return path

def run( videoIn, videoOut = '', frameOutDir = '', frameSamplingFreq = 0.):
	src = cv2.VideoCapture(videoIn)
	out = None
	tot = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(src.get(cv2.CAP_PROP_FPS))

	print("Processing %d frames in %s"%(tot, videoIn))
	if videoOut != '':
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(createDir(videoOut, False),fourcc, 60.02, (1920,1080))

	if frameOutDir != '' and frameSamplingFreq > 0:
		createDir(frameOutDir, True)
		frameSamplingFreq = min(frameSamplingFreq, fps)
		if videoIn is 0:
			framePrefix = 'camera'
		else:
			_, framePrefix = os.path.split(videoIn)
	else:
		frameSamplingFreq = 0

	##Define the intensity correction to tune the contrast
	lookUpTable = np.empty((1,256),np.uint8)
	for i in range(256):
		#lookUpTable[0,i] = np.clip(pow(i/255.0,0.5)*255.0,0,255) #gamma correction
		lookUpTable[0,i] = np.clip(0.4*i -4.8 + (158/(1+np.exp((19-i)/5.5))),0,255) #linear + sigmoid
		#print '%d -> %d' %(i,lookUpTable[0,i])  #Check if lookUpTable[0,0] = 0, lookUpTable[0,255] = 255
		
	##Skip some frames at the begining of the video which are taken when the camera was not stable
	#ret, frame = src.read()
	frameCnt = 0
	showLine = 0
	SkipFramesNum = 2*60*60 #Mins*seconds*fps
	for i in range(SkipFramesNum):
		ret, frame = src.read()
		frameCnt += 1
	
	if ret:
		frame = frame.astype("float")
		fore = frame.copy()
		back = frame.copy()
		#frameCnt = 0
		#showLine = 0

		while True:
			ret, frame = src.read()
			if not ret:
				break
			frame = frame.astype("float")
			frameCnt += 1

			fore = frame.copy()
			#cv2.accumulateWeighted(frame, fore, 0.1)
			cv2.accumulateWeighted(frame, back, 0.002)
			
			showDelta = cv2.absdiff(cv2.convertScaleAbs(fore),cv2.convertScaleAbs(back))
			showDelta = cv2.LUT(showDelta, lookUpTable)  #Use the defined intensity tuning
			
			if out:
				out.write(showDelta)
			if frameSamplingFreq and frameCnt % int(fps / frameSamplingFreq) is 0:
				frameCntStr = str(frameCnt)
				frameCntStr = frameCntStr.zfill(6)		#Fill zeros in filename
				cv2.imwrite(os.path.join(frameOutDir, "%s_" %(framePrefix) + frameCntStr+".png" ), showDelta)
			
			if frameCnt > showLine * tot:
				showLine += 0.1
				print(dt.now(), "...finished %2.0f%%"%(frameCnt*100.0 / tot))
		
	src.release()
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	for root, dirs, files in os.walk('/home/user/CloudChamber/clipstodo/'):
		for file in files:
			if file[-4:] == '.mp4':
				run( os.path.join(root, file), 
					 videoOut = os.path.join(root, file + '_deBkg.avi'), #videoOut = ''
					 frameOutDir = os.path.join(root, file + '_frames'), 
					 frameSamplingFreq = 2)
