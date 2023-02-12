#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <SFML/Graphics/Image.hpp>
#include "FiltersProvider.hpp"

const std::string IMAGE_NAME = "test2";
const std::string INPUT_IMAGE_NAME = IMAGE_NAME + ".jpg";
const std::string OUTPUT_IMAGE_NAME = IMAGE_NAME + "_out.jpg";
const std::string OUTPUT_IMAGE_NAME_EDGE_DETECT = IMAGE_NAME + "_edgeDetect.jpg";
std::vector<std::string> filterNames = { "Blur", "Sharpen", "EdgeDetection", "Emboss", "Outline" };
using KernelRow = std::vector<float>;
using Kernel = std::vector<KernelRow>;

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
    const auto start = std::chrono::high_resolution_clock::now();
    std::forward<Callable>(function)(std::forward<Args>(params)...);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    return duration;
}

auto loadImage()
{
    sf::Image image{}, dest{};
    image.loadFromFile("../images/" + INPUT_IMAGE_NAME);
    return image;
}
void saveImgEdgeDetect(sf::Image& image)
{
    image.saveToFile("../images/" + OUTPUT_IMAGE_NAME_EDGE_DETECT);
}

void saveImage(sf::Image& image)
{
    image.saveToFile("../images/" + OUTPUT_IMAGE_NAME);
}

void alignChannel(int& channelValue)
{
    channelValue = (channelValue > 255) ? 255 : channelValue;
    channelValue = (channelValue < 0) ? 0 : channelValue;
}

void applyFilter(sf::Image& image, const Kernel& filter)
{
    const auto kernelSize = static_cast<int>(filter.size());
    const auto kernelMargin = kernelSize / 2;
    const auto imageHeight = static_cast<int>(image.getSize().y);
    const auto imageWidth = static_cast<int>(image.getSize().x);
    auto outputImage = image;

    for (int y = kernelMargin; y < imageHeight - kernelMargin; ++y)
    {
        for (int x = kernelMargin; x < imageWidth - kernelMargin; ++x)
        {
            int newRedChannel{}, newGreenChannel{}, newBlueChannel{};
            for (int kernelX = -kernelMargin; kernelX <= kernelMargin; ++kernelX)
            {
                for (int kernelY = -kernelMargin; kernelY <= kernelMargin; ++kernelY)
                {
                    const auto kernelValue = filter[kernelX + kernelMargin][kernelY + kernelMargin];
                    const auto pixel = image.getPixel(x + kernelX, y + kernelY);
                    newRedChannel += static_cast<int>(pixel.r * kernelValue);
                    newGreenChannel += static_cast<int>(pixel.g * kernelValue);
                    newBlueChannel += static_cast<int>(pixel.b * kernelValue);
                }
            }

            alignChannel(newRedChannel);
            alignChannel(newGreenChannel);
            alignChannel(newBlueChannel);
            outputImage.setPixel(x, y, sf::Color(newRedChannel, newGreenChannel, newBlueChannel));
        }
    }

    image = std::move(outputImage);
}

int main()
{
    auto image = loadImage();

    // Apply each of the 5 filters and save the output image
    auto edgeDetectionFilter = Filter::edgeDetectionKernel();
    auto edgeDetectionImage = image;
    const auto edgeDetectionDuration = runWithTimeMeasurementCpu(applyFilter, edgeDetectionImage, edgeDetectionFilter);
    std::cout << "Duration for Edge Detection [ms]: " << edgeDetectionDuration << std::endl;
    edgeDetectionImage.saveToFile("../images/" + IMAGE_NAME + "_edge_detection.jpg");

    auto blurFilter = Filter::blurKernel();
    auto blurImage = image;
    const auto blurDuration = runWithTimeMeasurementCpu(applyFilter, blurImage, blurFilter);
    std::cout << "Duration for Blur [ms]: " << blurDuration << std::endl;
    blurImage.saveToFile("../images/" + IMAGE_NAME + "_blur.jpg");

    auto sharpenFilter = Filter::sharpenKernel();
    auto sharpenImage = image;
    const auto sharpenDuration = runWithTimeMeasurementCpu(applyFilter, sharpenImage, sharpenFilter);
    std::cout << "Duration for Sharpen [ms]: " << sharpenDuration << std::endl;
    sharpenImage.saveToFile("../images/" + IMAGE_NAME + "_sharpen.jpg");

    auto embossFilter = Filter::embossKernel();
    auto embossImage = image;
    const auto embossDuration = runWithTimeMeasurementCpu(applyFilter, embossImage, embossFilter);
    std::cout << "Duration for Emboss [ms]: " << embossDuration << std::endl;
    embossImage.saveToFile("../images/" + IMAGE_NAME + "_emboss.jpg");

    auto outlineFilter = Filter::outlineKernel();
    auto outlineImage = image;
    const auto boxBlurDuration = runWithTimeMeasurementCpu(applyFilter, outlineImage, outlineFilter);
    std::cout << "Duration for Outline [ms]: " << boxBlurDuration << std::endl;
    outlineImage.saveToFile("../images/" + IMAGE_NAME + "_outline.jpg");

    return EXIT_SUCCESS;
}

