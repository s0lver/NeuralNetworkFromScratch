package com.s0lver;

import java.util.Random;

public class Utils {
    static Random random = new Random(0);

    public static double generateRandomDouble(double min, double max) {
        //        double a = random.nextDouble();
        //        double num = min + random.nextDouble() * (max - min);
        //        if (a < 0.5)
        //            return num;
        //        else
        //            return -num;

        return min + random.nextDouble() * (max - min);
    }

    public static double Sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }
}
