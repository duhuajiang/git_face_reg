
#include "Dect_face.hpp"


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
