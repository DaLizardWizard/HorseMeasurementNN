# HorseMeasurementNN
This is the pipeline for classifying horse images into body condition score, segmenting the image and measuring body geometry. it is built using two neural nets and one computer vision pipeline(s)

NN1 - Extraction of horse via image segmentation
1) using DLAB NN trained on COCO, images containing horse are marked
2) Any non-horse pixels are removed to produce a refined horse image

NN2 - Classification of horse body type
1) Images are scraped from the web based upon certain keywords
2) Images are passed through NN1 to remove any non-horse pixels from images.
3) A NN classifier is trained on the various image categories

CV1 - Analysis of new images and geeometry determination
1) Images are passed through NN1 to remove any non-horse pixels
2) Images are passed through NN2 to determine body category
3) Markers on the horse are used to determine height
