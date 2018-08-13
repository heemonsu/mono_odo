#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;
using namespace xfeatures2d;


/////////////////////////////////////////////////////////////- ELLIPTICAL KEYPOINTS CONVERTION FOR BECHMARKING -/////////////////////////////////////////////////////////////

template<typename _Tp> static int solveQuadratic(_Tp a, _Tp b, _Tp c, _Tp& x1, _Tp& x2)
{
    if( a == 0 )
    {
        if( b == 0 )
        {
            x1 = x2 = 0;
            return c == 0;
        }
        x1 = x2 = -c/b;
        return 1;
    }

    _Tp d = b*b - 4*a*c;
    if( d < 0 )
    {
        x1 = x2 = 0;
        return 0;
    }
    if( d > 0 )
    {
        d = std::sqrt(d);
        double s = 1/(2*a);
        x1 = (-b - d)*s;
        x2 = (-b + d)*s;
        if( x1 > x2 )
            std::swap(x1, x2);
        return 2;
    }
    x1 = x2 = -b/(2*a);
    return 1;
}

class EllipticKeyPoint
{
public:
    EllipticKeyPoint();
    EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse );

    static void convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst );
    static void convert( const std::vector<EllipticKeyPoint>& src, std::vector<KeyPoint>& dst );

    static Mat_<double> getSecondMomentsMatrix( const Scalar& _ellipse );
    Mat_<double> getSecondMomentsMatrix() const;

    void calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const;
    static void calcProjection( const std::vector<EllipticKeyPoint>& src, const Mat_<double>& H, std::vector<EllipticKeyPoint>& dst );

    Point2f center;
    Scalar ellipse; // 3 elements a, b, c: ax^2+2bxy+cy^2=1
    Size_<float> axes; // half length of ellipse axes
    Size_<float> boundingBox; // half sizes of bounding box which sides are parallel to the coordinate axes
};

EllipticKeyPoint::EllipticKeyPoint()
{
    *this = EllipticKeyPoint(Point2f(0,0), Scalar(1, 0, 1) );
}

EllipticKeyPoint::EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse )
{
    center = _center;
    ellipse = _ellipse;

    double a = ellipse[0], b = ellipse[1], c = ellipse[2];
    double ac_b2 = a*c - b*b;
    double x1, x2;
    solveQuadratic(1., -(a+c), ac_b2, x1, x2);
    axes.width = (float)(1/sqrt(x1));
    axes.height = (float)(1/sqrt(x2));

    boundingBox.width = (float)sqrt(ellipse[2]/ac_b2);
    boundingBox.height = (float)sqrt(ellipse[0]/ac_b2);
}

void EllipticKeyPoint::convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst )
{
    //CV_INSTRUMENT_REGION()

    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            float rad = src[i].size/2;
            CV_Assert( rad );
            float fac = 1.f/(rad*rad);
            dst[i] = EllipticKeyPoint( src[i].pt, Scalar(fac, 0, fac) );
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
