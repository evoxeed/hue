package org.example;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws IOException {
        // Загружаем изображение
        Mat originalImage = Imgcodecs.imread("input/cat.jpeg");

        // Создаем копию изображения и конвертируем его в HSV.
        Mat hsvImage = originalImage.clone();
        Imgproc.cvtColor(hsvImage, hsvImage, Imgproc.COLOR_BGR2HSV);

        // Создаем копию изображения и конвертируем его в градации серого.
        Mat grayImage = originalImage.clone();
        Imgproc.cvtColor(grayImage, grayImage, Imgproc.COLOR_BGR2GRAY);

        // Разделяем каналы HSV
        List<Mat> hsvPlanes = new ArrayList<>();
        Core.split(hsvImage, hsvPlanes);

        // Гистограмма для hue
        Mat hueHist = new Mat();
        Imgproc.calcHist(Collections.singletonList(hsvPlanes.get(0)), new MatOfInt(0), new Mat(), hueHist, new MatOfInt(180), new MatOfFloat(0, 179));

        // Создаем изображение гистограммы
        int bins = 180;
        double[] histData = new double[bins];
        for (int i = 0; i < bins; i++) {
            histData[i] = (double) hueHist.get(i, 0)[0];
        }

        // Создаем набор данных для гистограммы
        HistogramDataset dataset = new HistogramDataset();
        dataset.addSeries("Hue", histData, bins);

        // Создаем график гистограммы
        JFreeChart histogram = ChartFactory.createHistogram(
                "График гистограммы",
                "",
                "",
                dataset
        );

        // Записываем гистограмму в файл
        ChartUtils.saveChartAsPNG(new File("result/histogram.png"), histogram, 800, 600);

        // Находим максимум в гистограмме
        Core.MinMaxLocResult histMaxLoc = Core.minMaxLoc(hueHist);

        // Готовим изображение для вывода
        Mat outputImage = originalImage.clone();

        // Пороговое значение разницы в hue
        double hueDifferenceThreshold = 5;

        // Проходимся по изображению и меняем значения пикселей
        for (int x = 0; x < hsvImage.rows(); x++) {
            for (int y = 0; y < hsvImage.cols(); y++) {
                double[] hsvPixel = hsvImage.get(x, y);
                double[] grayPixel = grayImage.get(x, y);

                if (Math.abs(hsvPixel[0] - histMaxLoc.maxLoc.y) > hueDifferenceThreshold) {
                    // Задаем пиксель как серый.
                    outputImage.put(x, y, grayPixel[0], grayPixel[0], grayPixel[0]);
                }
            }
        }

        Mat originalWithBorder = new Mat();
        Mat outputWithBorder = new Mat();
        Core.copyMakeBorder(originalImage, originalWithBorder, 0, 100, 0, 0, Core.BORDER_CONSTANT, new Scalar(255,255,255));
        Core.copyMakeBorder(outputImage, outputWithBorder, 0, 100, 0, 0, Core.BORDER_CONSTANT, new Scalar(255,255,255));

        // Вычисляем размеры текста
        Size textSizeOriginal = Imgproc.getTextSize("Оригинал", Imgproc.FONT_HERSHEY_COMPLEX, 2.0, 2, null);
        Size textSizeResult = Imgproc.getTextSize("Результат", Imgproc.FONT_HERSHEY_COMPLEX, 2.0, 2, null);

        // Вычисляем положение текста, чтобы он был расположен по центру
        Point textPositionOriginal = new Point(
                (originalWithBorder.cols() - textSizeOriginal.width) / 2,
                originalWithBorder.rows() - textSizeOriginal.height
        );
        Point textPositionResult = new Point(
                (outputWithBorder.cols() - textSizeResult.width) / 2,
                outputWithBorder.rows() - textSizeResult.height
        );

        // Добавляем текст в новую область под каждым изображением
        Imgproc.putText(originalWithBorder, "Оригинал", textPositionOriginal, Imgproc.FONT_HERSHEY_COMPLEX, 2.0, new Scalar(0,0,0), 2);
        Imgproc.putText(outputWithBorder, "Результат", textPositionResult, Imgproc.FONT_HERSHEY_COMPLEX, 2.0, new Scalar(0,0,0), 2);

        // Теперь интегрируем оба обработанных изображения
        List<Mat> images = new ArrayList<>();
        images.add(originalWithBorder);
        images.add(outputWithBorder);
        Mat combinedImage = new Mat();
        Core.hconcat(images, combinedImage);

        // Сохраняем изображение
        Imgcodecs.imwrite("result/Sravnenie.jpg", combinedImage);

        // Сохраняем изображение рузультата
         Imgcodecs.imwrite("result/Result.jpg", outputImage);
    }
}