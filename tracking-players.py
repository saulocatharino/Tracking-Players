### TRACKING-PLAYERS ###

# importando os pacotes necessários
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())


OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# inicializar o rastreador
trackers = cv2.MultiTracker_create()

if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

else:
	vs = cv2.VideoCapture(args["video"])

# loop dos frames do vídeo
while True:
	# pegar o frame atual e manipular se estiver usando um objeto videoStream ou VideoCapture.
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# redimensionar o frame
	frame = imutils.resize(frame, width=1200)

	# pegar as coordenadas da caixa atualizadas (se houver) para cada uma.
	# objeto que está sendo rastreado
	(success, boxes) = trackers.update(frame)

	# loop sobre as boxs
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# caixa para rastrear
	if key == ord("s"):
		# selecionar a box do objeto que queremos rastrear (pressionando ENTER depois de selecionar a ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# criar um novo rastreador do objeto para a box a adicionar
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)

	elif key == ord("q"):
		break

if not args.get("video", False):
	vs.stop()

else:
	vs.release()

cv2.destroyAllWindows()