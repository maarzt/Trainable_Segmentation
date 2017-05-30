package trainableSegmentation.ij2;

import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.outofbounds.OutOfBoundsFactory;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.view.Views;
import net.imglib2.view.composite.RealComposite;
import org.scijava.Context;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Matthias Arzt
 */
public class FeatureStack2 {

	private static OpService ops = new Context(OpService.class).getService(OpService.class);

	public static RandomAccessibleInterval<FloatType> createGaussStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		final double minimumSigma = 1;
		final double maximumSigma = 16;
		for(double sigma = minimumSigma; sigma <= maximumSigma; sigma *= 2)
			features.add(ops.filter().gauss(image, sigma*0.4));
		return Views.stack(features);
	}

	public static RandomAccessibleInterval<FloatType> createHessianStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(Views.stack(Collections.singletonList(image)));
		final double minimumSigma = 1;
		final double maximumSigma = 16;
		features.add(calculateHessianOnChannel(image, 0));
		for(double sigma = minimumSigma; sigma <= maximumSigma; sigma *= 2)
			features.add(calculateHessianOnChannel(image, sigma));
		return Views.concatenate(2, features);
	}

	public static RandomAccessibleInterval<FloatType> calculateHessianOnChannel(Img<FloatType> image, double sigma) {
		double[] sigmas = {0.4 * sigma, 0.4 * sigma};

		RandomAccessibleInterval<FloatType> blurred = gauss(image, sigmas);
		RandomAccessibleInterval<FloatType> dx = deriveX(blurred);
		RandomAccessibleInterval<FloatType> dy = deriveY(blurred);
		RandomAccess<FloatType> dxx = deriveX(dx).randomAccess();
		RandomAccess<FloatType> dxy = deriveY(dx).randomAccess();
		RandomAccess<FloatType> dyy = deriveY(dy).randomAccess();

		Img<FloatType> features = ops.create().img(extendDimension(image, 8), new FloatType());

		Cursor<RealComposite<FloatType>> cursor = Views.iterable(Views.collapseReal(features)).cursor();
		while (cursor.hasNext()) {
			cursor.next();
			dxx.setPosition(cursor);
			dxy.setPosition(cursor);
			dyy.setPosition(cursor);
			float s_xx = dxx.get().get();
			float s_xy = dxy.get().get();
			float s_yy = dyy.get().get();
			calculateHessianPerPixel(cursor.get(), s_xx, s_xy, s_yy);
		}

		return features;
	}
	
	public static RandomAccessibleInterval<FloatType>
		createDifferenceOfGaussiansStack(Img<FloatType> image)
	{
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		final double minimumSigma = 1;
		final double maximumSigma = 16;
		for (double sigma1 = minimumSigma; sigma1 <= maximumSigma; sigma1 *= 2)
			for (double sigma2 = minimumSigma; sigma2 < sigma1; sigma2 *= 2)
				features.add(calculateDifferenceOfGaussians(image, sigma1, sigma2));
		return Views.stack(features);
	}

	private static Dimensions extendDimension(Dimensions dimension, int size) {
		int numDimensions = dimension.numDimensions();
		long[] result = new long[numDimensions + 1];
		for (int j = 0; j < numDimensions; j++) result[j] = dimension.dimension(j);
		result[numDimensions] = size;
		return new FinalDimensions(result);
	}

	private static final int HESSIAN = 0;
	private static final int TRACE = 1;
	private static final int DETERMINANT = 2;
	private static final int EIGENVALUE_1 = 3;
	private static final int EIGENVALUE_2 = 4;
	private static final int ORIENTATION = 5;
	private static final int SQUARE_EIGENVALUE_DIFFERENCE = 6;
	private static final int NORMALIZED_EIGENVALUE_DIFFERENCE = 7;

	public static void calculateHessianPerPixel(RealComposite<FloatType> output,
												float s_xx, float s_xy, float s_yy)
	{
		final double t = Math.pow(1, 0.75);

		// Hessian module: sqrt (a^2 + b*c + d^2)
		output.get(HESSIAN).set((float) Math.sqrt(s_xx*s_xx + s_xy*s_xy+ s_yy*s_yy));
		// Trace: a + d
		final float trace = s_xx + s_yy;
		output.get(TRACE).set(trace);
		// Determinant: a*d - c*b
		final float determinant = s_xx*s_yy-s_xy*s_xy;
		output.get(DETERMINANT).set(determinant);

		// Ratio
		//ipRatio.set((float)(trace*trace) / determinant);
		// First eigenvalue: (a + d) / 2 + sqrt( ( 4*b^2 + (a - d)^2) / 2 )
		output.get(EIGENVALUE_1).set((float) ( trace/2.0 + Math.sqrt((4*s_xy*s_xy + (s_xx - s_yy)*(s_xx - s_yy)) / 2.0 ) ) );
		// Second eigenvalue: (a + d) / 2 - sqrt( ( 4*b^2 + (a - d)^2) / 2 )
		output.get(EIGENVALUE_2).set((float) ( trace/2.0 - Math.sqrt((4*s_xy*s_xy + (s_xx - s_yy)*(s_xx - s_yy)) / 2.0 ) ) );
		// Orientation
		if (s_xy < 0.0) // -0.5 * acos( (a-d) / sqrt( 4*b^2 + (a - d)^2)) )
		{
			float orientation =(float)( -0.5 * Math.acos((s_xx	- s_yy)
					/ Math.sqrt(4.0 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) ));
			if (Float.isNaN(orientation))
				orientation = 0;
			output.get(ORIENTATION).set(orientation);
		}
		else 	// 0.5 * acos( (a-d) / sqrt( 4*b^2 + (a - d)^2)) )
		{
			float orientation =(float)( 0.5 * Math.acos((s_xx	- s_yy)
					/ Math.sqrt(4.0 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) ));
			if (Float.isNaN(orientation))
				orientation = 0;
			output.get(ORIENTATION).set(orientation);
		}
		// Gamma-normalized square eigenvalue difference
		output.get(SQUARE_EIGENVALUE_DIFFERENCE).set((float) ( Math.pow(t,4) * trace*trace * ( (s_xx - s_yy)*(s_xx - s_yy) + 4*s_xy*s_xy ) ) );
		// Square of Gamma-normalized eigenvalue difference
		output.get(NORMALIZED_EIGENVALUE_DIFFERENCE).set((float) ( Math.pow(t,2) * ( (s_xx - s_yy)*(s_xx - s_yy) + 4*s_xy*s_xy ) ) );
	}
	public static RandomAccessibleInterval<FloatType> gauss(Img<FloatType> image, double[] sigmas) {
		RandomAccessibleInterval<FloatType> blurred = ops.create().img(image);
		ops.filter().gauss(blurred, image, sigmas, new OutOfBoundsBorderFactory<>());
		return blurred;
	}

	private static final float[] SOBEL_FILTER_X_VALUES = {1f,2f,1f,0f,0f,0f,-1f,-2f,-1f};
	private static final float[] SOBEL_FILTER_Y_VALUES = {1f,0f,-1f,2f,0f,-2f,1f,0f,-1f};

	private static final RandomAccessibleInterval<FloatType> SOBEL_FILTER_X = ArrayImgs.floats(SOBEL_FILTER_X_VALUES, 3, 3);
	private static final RandomAccessibleInterval<FloatType> SOBEL_FILTER_Y = ArrayImgs.floats(SOBEL_FILTER_Y_VALUES, 3, 3);

	public static RandomAccessibleInterval<FloatType> deriveX(RandomAccessibleInterval<FloatType> in) {
		return convolve(in, SOBEL_FILTER_X);
	}

	public static RandomAccessibleInterval<FloatType> deriveY(RandomAccessibleInterval<FloatType> in) {
		return convolve(in, SOBEL_FILTER_Y);
	}

	public static RandomAccessibleInterval<FloatType> convolve(RandomAccessibleInterval<FloatType> blurred, RandomAccessibleInterval<FloatType> kernel) {
		return ops.filter().convolve(blurred, kernel, new OutOfBoundsBorderFactory<>());
	}

	public static RandomAccessibleInterval<FloatType> calculateDifferenceOfGaussians(Img<FloatType> img, double sigma1, double sigma2) {
		RandomAccessibleInterval<FloatType> result = ops.create().img(img);
		OutOfBoundsFactory<FloatType, RandomAccessibleInterval<FloatType>> border = new OutOfBoundsBorderFactory<>();
		ops.filter().dog(result, img, 0.4 * sigma1, 0.4 * sigma2, border);
		return result;
	}

	public static RandomAccessibleInterval<FloatType> createSobelStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		final double minimumSigma = 1;
		final double maximumSigma = 16;
		features.add(calculateSingleGradient(image, 0.0));
		for(double sigma = minimumSigma; sigma <= maximumSigma; sigma *= 2)
			features.add(calculateSingleGradient(image, sigma));
		return Views.stack(features);
	}

	private static RandomAccessibleInterval<FloatType> calculateSingleGradient(Img<FloatType> image, double sigma) {
		RandomAccessibleInterval<FloatType> blurred = gauss(image, new double[]{0.4 * sigma, 0.4 * sigma});
		RandomAccessibleInterval<Pair<FloatType, FloatType>> derivatives =
				Views.interval(Views.pair(deriveX(blurred), deriveY(blurred)), blurred);
		Converter<Pair<FloatType, FloatType>, FloatType> converter =
			(input, output) -> output.set(norm2(input.getA().get(), input.getB().get()));
		return Converters.convert(derivatives, converter, new FloatType());
	}

	private static float norm2(float x, float y) {
		return (float) Math.sqrt(x * x + y * y);
	}

	public static RandomAccessibleInterval<FloatType> createLipschitzStack(Img<FloatType> image) {
		List<RandomAccessibleInterval<FloatType>> features = new ArrayList<>();
		features.add(image);
		for(double slope = 5; slope < 30; slope += 5)
			features.add(calculateLipschitz(image, true, true, slope));
		return Views.stack(features);
	}

	public static RandomAccessibleInterval<FloatType> calculateLipschitz(RandomAccessibleInterval<FloatType> image, boolean downHat, boolean topHat, double slope) {
		Lipschitz2d lipschitz = new Lipschitz2d(topHat, downHat, slope);
		lipschitz.ops = ops;
		return toFloat(lipschitz.calculate(toInt(image)));
	}

	private static RandomAccessibleInterval<IntType> toInt(RandomAccessibleInterval<FloatType> input) {
		Converter<FloatType, IntType> floatToInt = (in, out) -> out.set((int) in.get());
		return Converters.convert(input, floatToInt, new IntType());
	}

	private static RandomAccessibleInterval<FloatType> toFloat(RandomAccessibleInterval<IntType> input) {
		Converter<IntType, FloatType> intToFloat = (in, out) -> out.set(in.get());
		return Converters.convert(input, intToFloat, new FloatType());
	}
}
