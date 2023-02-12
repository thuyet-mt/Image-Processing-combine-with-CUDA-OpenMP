#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <SFML/Graphics/Image.hpp>
#include <omp.h>
#include "FiltersProvider.hpp"

const std::string IMAGE_NAME = "test2";
const std::string INPUT_IMAGE_NAME = IMAGE_NAME + ".jpg";
const std::string OUTPUT_IMAGE_NAME = IMAGE_NAME + "_out.jpg";

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

void saveImage(sf::Image& image)
{
    image.saveToFile("../images/" + OUTPUT_IMAGE_NAME);
}

void alignChannel(int& channelValue)
{
    channelValue = (channelValue > 255) ? 255 : channelValue;
    channelValue = (channelValue < 0) ? 0 : channelValue;
}

void applyFilter(sf::Image& image, const Filter::Kernel& filter)
{
    const auto kernelSize = static_cast<int>(filter.size());
    const auto kernelMargin = kernelSize / 2;
    const auto imageHeight = static_cast<int>(image.getSize().y);
    const auto imageWidth = static_cast<int>(image.getSize().x);
    auto outputImage = image;

    int x, y;

    #pragma omp parallel for private(x, y) schedule(dynamic)
    for (y = kernelMargin; y < imageHeight - kernelMargin; ++y)
    {
        for (x = kernelMargin; x < imageWidth - kernelMargin; ++x)
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
    omp_set_num_threads(omp_get_max_threads());
    auto image = loadImage();

    // Apply each of the 5 filters and save the output image
    auto edgeDetectionFilter = Filter::edgeDetectionKernel();
    auto edgeDetectionImage = image;
    const auto edgeDetectionDuration = runWithTimeMeasurementCpu(applyFilter, edgeDetectionImage, edgeDetectionFilter);
    std::cout << "Duration for Edge Detection OMP [ms]: " << edgeDetectionDuration << std::endl;
    edgeDetectionImage.saveToFile("../images/" + IMAGE_NAME + "_edge_detection_omp.jpg");

    auto blurFilter = Filter::blurKernel();
    auto blurImage = image;
    const auto blurDuration = runWithTimeMeasurementCpu(applyFilter, blurImage, blurFilter);
    std::cout << "Duration for Blur OMP [ms]: " << blurDuration << std::endl;
    blurImage.saveToFile("../images/" + IMAGE_NAME + "_blur_omp.jpg");

    auto sharpenFilter = Filter::sharpenKernel();
    auto sharpenImage = image;
    const auto sharpenDuration = runWithTimeMeasurementCpu(applyFilter, sharpenImage, sharpenFilter);
    std::cout << "Duration for Sharpen OMP [ms]: " << sharpenDuration << std::endl;
    sharpenImage.saveToFile("../images/" + IMAGE_NAME + "_sharpen_omp.jpg");

    auto embossFilter = Filter::embossKernel();
    auto embossImage = image;
    const auto embossDuration = runWithTimeMeasurementCpu(applyFilter, embossImage, embossFilter);
    std::cout << "Duration for Emboss OMP [ms]: " << embossDuration << std::endl;
    embossImage.saveToFile("../images/" + IMAGE_NAME + "_emboss_omp.jpg");

    auto outlineFilter = Filter::outlineKernel();
    auto outlineImage = image;
    const auto boxBlurDuration = runWithTimeMeasurementCpu(applyFilter, outlineImage, outlineFilter);
    std::cout << "Duration for Outline OMP [ms]: " << boxBlurDuration << std::endl;
    outlineImage.saveToFile("../images/" + IMAGE_NAME + "_outline_omp.jpg");

    return EXIT_SUCCESS;
}
