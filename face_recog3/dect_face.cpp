
/*#include "Dect_face.hpp"


Face detect_face(Mat image,CascadeClassifier ccf)
{
        vector<Rect> faces;
        Mat gray;
        cvtColor(image,gray,CV_BGR2GRAY);
        equalizeHist(gray,gray);
        ccf.detectMultiScale(gray,faces,1.1,4,0,Size(120,120),Size(500,500));

        vector<Mat>  face_img;

        for(size_t i=0;i<faces.size();i++)
        {
            Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            face_img.push_back(image(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height)));
        }
        Face f_re={.faces = faces,.face_img = face_img};
        return  f_re;
}


Face detect_face1(Mat frame,dnn::Net net)
{

        float min_confidence = 0.5;
        if (frame.channels() == 4)cvtColor(frame, frame, COLOR_BGRA2BGR);

        Mat inputBlob = blobFromImage(frame, inScaleFactor,Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images

        net.setInput(inputBlob, "data");
        Mat detection = net.forward("detection_out");
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        float confidenceThreshold = min_confidence;
        for (int i = 0; i < detectionMat.rows; i++)
            {
                float confidence = detectionMat.at<float>(i, 2);

                if (confidence > confidenceThreshold)
                {
                    int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                    int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                    int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                    int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                    Rect object((int)xLeftBottom, (int)yLeftBottom,(int)(xRightTop - xLeftBottom),
                                (int)(yRightTop - yLeftBottom));

                    rectangle(frame, object, Scalar(0, 255, 0));
                    Mat tmp = frame(object);


                }
            }

        }
        return 0;

}

*/

#include "Dect_face.hpp"


Dector::Dector(const string& face_prototxt,const string& face_model,const float confidence,const int inWidth,const int inHeight)
{
        net = dnn::readNetFromCaffe(face_prototxt, face_model);
        min_confidence = confidence;
        face_inWidth = inWidth;
        face_inHeight = inHeight;

}

Face Dector::detect_face(Mat frame)
{
    const Scalar meanVal(104.0, 177.0, 123.0);
    const double inScaleFactor = 1.0;
    if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);
    Mat inputBlob = dnn::blobFromImage(frame, inScaleFactor,Size(face_inWidth, face_inHeight), meanVal, false, false);

    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


    vector<Rect> faces;
    vector<Mat>  face_img;


    for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > min_confidence)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

                Mat tmp = frame(object);
                face_img.push_back(tmp);
                faces.push_back(object);


            }

        }
     Face f_re={.faces = faces,.face_img = face_img};
     return  f_re;
}


