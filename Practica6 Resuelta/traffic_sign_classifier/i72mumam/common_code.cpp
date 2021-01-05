#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "common_code.hpp"
#include "lbp.hpp"

#ifndef NDEBUG
#include<opencv2/highgui.hpp>

const std::string __Debug_Padding =
"                                                                              "
"                                                                              "
"                                                                              "
"                                                                              "
"                                                                              ";
size_t __Debug_PPL = 3;
#endif

cv::Mat
fsiv_normalize_mean_var(cv::Mat const& src)
{
    assert(!src.empty() && src.channels()==1);
    cv::Mat dst;

    //TODO: normalize source image so its mean will be equal to 0.0 and
    // its var equal to 1.0.
    // Hint: use cv::meanStdDev() to get the source mean and stdev.

    cv::normalize(src, dst, 0.0, 1.0);

    //
#ifndef NDEBUG
    assert(!dst.empty());
    {
        std::vector<double> mean, stdev;
        cv::meanStdDev(dst, mean, stdev);
        assert(std::abs(mean[0])<=1.0e-5 && std::abs(stdev[0]-1.0)<=1.0e-5);
    }
#endif
    return dst;
}

cv::Mat
fsiv_normalize_minmax(cv::Mat const& src)
{
    assert(!src.empty() && src.channels()==1);
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);

    //TODO: normalize the source image so its mininum value will be 0.0 and its
    // maximun value be 1.0
    // Hint: use cv::normalize()

     // std::cout<<src<<std::endl;
    
    cv::normalize(src, dst, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);


    //

     // std::cout<<dst<<std::endl;
#ifndef NDEBUG
    assert(!dst.empty());
    {
        assert(dst.type()==CV_32FC1);
        double min,max;
        cv::minMaxLoc(dst, &min, &max);

        assert(std::abs(min)<=1.0e-6 && std::abs(max-1.0)<=1.0e-6);
    }
#endif
    return dst;
}

bool
string_to_ncells(const std::string& str, int n_cells[])
{
    bool ret_val = true;
    std::istringstream input(str);
    char sep;
    input >> n_cells[0] >> sep >> n_cells[1];
    if (!input)
        ret_val = false;
    else
        ret_val = ((n_cells[0]*n_cells[1]) > 0);
    return ret_val;
}

bool
load_gtsrb_categories(const std::string& dataset_pathname,
                      std::vector<int>& cats,
                      std::vector<std::string>& descs)
{
    bool ret_val = false;
    std::ifstream in(dataset_pathname+"/train/CLASSES.TXT");
    if (in)
    {
        int class_label = 0;
        char sep = ' ';
        std::string class_description;
        while(in)
        {
            in >> class_label >> sep >> sep >> sep >> class_description;
            if (in)
            {
                cats.push_back(class_label);
                descs.push_back(class_description);
            }
        }
        ret_val = true;
    }
    return ret_val;
}

bool
load_gtsrb_train_metadata(const std::string& metadata_file,
                          std::vector<std::string> & lfiles,
                          std::vector<cv::Rect>& rois,
                          cv::Mat & labels)
{
    bool ret_val = true;
    DEBUG(1, "Loading metadata from file '"<< metadata_file << "'." << std::endl);
    std::ifstream input (metadata_file);
    if (input)
    {
        std::vector<int> labels_v;
        std::istringstream line;
        std::string dataset_path;

        size_t pos = metadata_file.rfind("/");
        if (pos != std::string::npos)
            dataset_path=metadata_file.substr(0, pos);
        else
            dataset_path=".";
        DEBUG(2, "\tDataset path is: "<< dataset_path << "'." << std::endl);
        while(input && ret_val)
        {
            std::string buffer;
            //get a line.
            input >> buffer;
            if (input)
            {
                DEBUG(3,"\tDecoding line: '"<< buffer << "'."<< std::endl);
                //replace ; by ' '
                for(size_t i=0;i<buffer.size(); ++i)
                    if (buffer[i]==';')
                        buffer[i]=' ';
                //get the line's metadata.
                std::istringstream line (buffer);
                std::string filename;
                int w, h, x1, y1, x2, y2, label;
                line >> filename >> w >> h >> x1 >> y1 >> x2 >> y2 >> label;
                if (line)
                {
                    lfiles.push_back(dataset_path+'/'+filename);
                    rois.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                    labels_v.push_back(label);
                    DEBUG(3,"\tDecoded fname: '" << lfiles.back() << "'." << std::endl);
                    DEBUG(3,"\tDecoded roi  :  " << rois.back() << std::endl);
                    DEBUG(3,"\tDecoded label:  " << labels_v.back() << std::endl);
                }
                else
                    ret_val = false;
            }
        }
        if (ret_val)
        {
            //Transform vector to cv::Mat.
            labels = cv::Mat(labels_v.size(), 1, CV_32SC1);
            std::copy(labels_v.begin(), labels_v.end(), labels.begin<int>());
        }
    }
    else
        ret_val = false;
    return ret_val;
}

bool
load_gtsrb_test_metadata(const std::string& dataset_path,
                          std::vector<std::string> & lfiles,
                          std::vector<cv::Rect>& rois,
                          cv::Mat & labels)
{
    bool ret_val = true;
    std::string metadata_file = dataset_path + "/test/metadada.csv";
    DEBUG(1, "Loading metadata from file '"<< metadata_file << "'."
          << std::endl);
    std::ifstream input (metadata_file);
    if (input)
    {
        std::vector<int> labels_v;
        std::istringstream line;
        while(input && ret_val)
        {
            std::string buffer;
            //get a line.
            input >> buffer;
            if (input)
            {
                DEBUG(3,"\tDecoding line: '"<< buffer << "'."<< std::endl);
                //replace ; by ' '
                for(size_t i=0;i<buffer.size(); ++i)
                    if (buffer[i]==';')
                        buffer[i]=' ';
                //get the line's metadata.
                std::istringstream line (buffer);
                std::string filename;
                int w, h, x1, y1, x2, y2, label;
                line >> filename >> w >> h >> x1 >> y1 >> x2 >> y2 >> label;
                if (line)
                {
                    lfiles.push_back(dataset_path+"/test/"+filename);
                    rois.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                    labels_v.push_back(label);
                    DEBUG(3,"\tDecoded fname: '" << lfiles.back() << "'." << std::endl);
                    DEBUG(3,"\tDecoded roi  :  " << rois.back() << std::endl);
                    DEBUG(3,"\tDecoded label:  " << labels_v.back() << std::endl);
                }
                else
                    ret_val = false;
            }
        }
        if (ret_val)
        {
            //Transform vector to cv::Mat.
            labels = cv::Mat(labels_v.size(), 1, CV_32SC1);
            std::copy(labels_v.begin(), labels_v.end(), labels.begin<int>());
        }
    }
    else
        ret_val = false;
    return ret_val;
}

bool
compute_lbp_from_list(const std::vector<std::string> & lfiles,
                      const std::vector<cv::Rect>& rois,
                      const cv::Size& canonical_size,
                      cv::Mat & lbp_descs, const int * ncells,
                      const int img_norm, bool hist_norm, const bool uLBP)
{
    DEBUG(1, "Computing lbp descriptors from files" << std::endl);
    bool ret_val = true;
    for (size_t i =0; i < lfiles.size() && ret_val; i++)
    {
        DEBUG(2, "\t Processing image: '" << lfiles[i] << "'." << std::endl);
        cv::Mat image = cv::imread(lfiles[i], cv::IMREAD_GRAYSCALE);       
        if (!image.empty())
        {
            image.convertTo(image, CV_32F, 1.0/255.0, 0.0);
#ifndef NDEBUG
            if (__Debug_Level>=3)
            {
                cv::imshow("IMAGE", image);
                cv::imshow("ROI", image(rois[i]));
            }
#endif
            if (img_norm==1)
                image = fsiv_normalize_minmax(image);
            else if (img_norm==2)
                image = fsiv_normalize_mean_var(image);
            cv::Mat canonical_img;
            cv::resize(image(rois[i]), canonical_img, canonical_size);
            cv::Mat lbp_img;
            fsiv_lbp(canonical_img, lbp_img, uLBP);
#ifndef NDEBUG
            if (__Debug_Level>=3)
            {
                cv::imshow("LBP", lbp_img);
            }
#endif

            cv::Mat lbp_desc;
            fsiv_lbp_desc(lbp_img, lbp_desc, ncells, hist_norm, uLBP);

            if (i==0)
            {
                lbp_descs = cv::Mat(lfiles.size(), lbp_desc.cols, CV_32FC1);
                lbp_desc.copyTo(lbp_descs.row(0));
            }
            else
                lbp_desc.copyTo(lbp_descs.row(i));
#ifndef NDEBUG
            if (__Debug_Level>=3)
            {
                if ((cv::waitKey(0)&0xff)==27)
                    return false;
            }
#endif
        }
        else
            ret_val = false;
    }
#ifndef NDEBUG
            if (__Debug_Level>=3)
                cv::destroyAllWindows();
#endif
    return ret_val;
}



void
draw_histogram(cv::Mat const& h_,
               cv::Mat& img,
               cv::Scalar color)
{

    assert(!img.empty() && img.type()==CV_8UC3);
    assert(!h_.empty() && h_.type()==CV_32FC1);
    const int hist_w = img.cols;
    const int hist_h = img.rows;
    const int hist_size = std::max(h_.rows, h_.cols);
    const int bin_w = cvRound( (double) hist_w/hist_size );
    cv::Mat h;
    cv::normalize(h_, h, 0, hist_h, cv::NORM_MINMAX);
    DEBUG(3, "Drawing histogram. N bins=" << hist_size << " img binw= "
          << bin_w << std::endl  );
    for( int i = 1; i < hist_size; i++ )
    {
        cv::line(img, cv::Point( bin_w*(i-1), hist_h - cvRound(h.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(h.at<float>(i)) ),
              color, 2, 8, 0  );
    }
}

void
draw_histogram_descriptor(cv::Mat const& desc,
             cv::Mat & img,
             int ncells[],
             cv::Scalar color)
{
    DEBUG(1, "Drawing an histogram descriptor with grip shape RxC: " << ncells[0]
            << 'x' << ncells[1] << " on an image with size " << img.rows
            << 'x' << img.cols << std::endl);
    assert(!desc.empty() && desc.type()==CV_32FC1);
    assert(ncells[0]*ncells[1]>0);
    assert(desc.rows==1);
    assert((desc.cols%(ncells[0]*ncells[1])) == 0);
    assert(img.empty() || (img.type()==CV_8UC3 && img.cols > desc.cols/(ncells[0]*ncells[1])));
    const int n_cells = ncells[0]*ncells[1];
    const int h_size = desc.cols/n_cells;
    if (img.empty())
        img = cv::Mat::zeros(h_size*n_cells/2, h_size*n_cells, CV_8UC3);
    const int cell_height = cvRound((double)img.rows/ncells[0]);
    const int cell_width = cvRound((double)img.cols/ncells[1]);
    DEBUG(2, "Cells img size (WxH):" << cell_width << 'x' << cell_height
          << std::endl);
    DEBUG(2, "Hist size is " << h_size << std::endl);
    for (int row=0, i=0; row < ncells[0]; ++row)
        for (int col=0; col < ncells[1]; ++col, ++i)
        {
            DEBUG(3,"Draw desc interval " << cv::Rect(i*h_size, 0, h_size, 1)
                  << " in img cell " << cv::Rect(col*cell_width, row*cell_height,
                                                 cell_width, cell_height)
                  << std::endl);
            cv::Mat h = desc(cv::Rect(i*h_size, 0, h_size, 1));
            cv::Mat img_roi = img(cv::Rect(col*cell_width, row*cell_height,
                                           cell_width, cell_height));
            draw_histogram(h, img_roi, color);
        }
}

float
compute_file_size(std::string const& fname, const long units)
{
    float size = -1.0;
    std::ifstream file (fname);
    if (file)
    {
        file.seekg (0, file.end);
        long length = file.tellg();
        size = static_cast<float>(length) / static_cast<float>(units);
    }
    return size;
}
