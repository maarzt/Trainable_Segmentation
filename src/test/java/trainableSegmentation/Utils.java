package trainableSegmentation;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Pair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.junit.Assert;
import trainableSegmentation.ij2.FeatureStackTest;

import java.net.URL;
import java.util.NoSuchElementException;
import java.util.StringJoiner;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * @author Matthias Arzt
 */
public class Utils {

	public static void assertImagesEqual(ImagePlus expected, ImagePlus actual) {
		assertTrue(diffImagePlus(expected, actual) == 0);
	}

	private static int diffImagePlus(final ImagePlus a, final ImagePlus b) {
		final int[] dimsA = a.getDimensions(), dimsB = b.getDimensions();
		if (dimsA.length != dimsB.length) return dimsA.length - dimsB.length;
		for (int i = 0; i < dimsA.length; i++) {
			if (dimsA[i] != dimsB[i]) return dimsA[i] - dimsB[i];
		}
		int count = 0;
		final ImageStack stackA = a.getStack(), stackB = b.getStack();
		for (int slice = 1; slice <= stackA.getSize(); slice++) {
			count += diff( stackA.getProcessor( slice ), stackB.getProcessor( slice ) );
		}
		return count;
	}

	public static < T extends Type< T >> boolean equals(final RandomAccessibleInterval< ? > a,
		final IterableInterval< T > b)
    {
    	if(!Intervals.equals(a, b))
    		return false;
        // create a cursor that automatically localizes itself on every move
		System.out.println("check picture content.");
        Cursor< T > bCursor = b.localizingCursor();
        RandomAccess< ? > aRandomAccess = a.randomAccess();
        while ( bCursor.hasNext())
        {
            bCursor.fwd();
            aRandomAccess.setPosition(bCursor);
            if( ! bCursor.get().equals( aRandomAccess.get() ))
            	return false;
        }
        return true;
    }

	private static int diff(final ImageProcessor a, final ImageProcessor b) {
		int count = 0;
		final int width = a.getWidth(), height = a.getHeight();
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (a.getf(x, y) != b.getf(x, y)) count++;
			}
		}
		return count;
	}

	public static ImagePlus loadImagePlusFromResource(final String path) {
		final URL url = Utils.class.getResource("/" + path);
		if(url == null)
			throw new NoSuchElementException("file: " + path);
		if ("file".equals(url.getProtocol())) return new ImagePlus(url.getPath());
		return new ImagePlus(url.toString());
	}

	public static void saveImageToResouce(final ImagePlus image, final String path) {
		final URL url = Utils.class.getResource(path);
		IJ.save(image, url.getPath());
	}

	public static <A extends IntegerType<A>, B extends IntegerType<B>>
		void assertImagesEqual(final IterableInterval<A> a, final RandomAccessibleInterval<B> b) {
		assertTrue(Intervals.equals(a, b));
		System.out.println("check picture content.");
		Cursor< A > aCursor = a.localizingCursor();
		RandomAccess< B > bRandomAccess = b.randomAccess();
		while ( aCursor.hasNext())
		{
			aCursor.fwd();
			bRandomAccess.setPosition(aCursor);
			assertEquals( bRandomAccess.get().getInteger(), aCursor.get().getInteger());
		}
	}

	public static <A extends Type<A>>
	void assertImagesEqual(final RandomAccessibleInterval<A> a, final RandomAccessibleInterval<A> b) {
		assertTrue(Intervals.equals(a, b));
		System.out.println("check picture content.");
		IntervalView<Pair<A, A>> pairs = Views.interval(Views.pair(a, b), b);
		Cursor<Pair<A, A>> cursor = pairs.cursor();
		while(cursor.hasNext()) {
			Pair<A,A> p = cursor.next();
			boolean equal = p.getA().valueEquals(p.getB());
			if(!equal)
				Assert.fail("Pixel values not equal on coordinate " +
						positionString(cursor) + ", expected: "
						+ p.getA() + " actual: " + p.getB());
		}
	}

	public static void assertImagesEqual(final ImagePlus expected, final RandomAccessibleInterval<FloatType> actual) {
		assertImagesEqual(ImagePlusAdapter.convertFloat(expected), actual);
	}

	public static void assertImagesEqual(final ImageProcessor expected, final RandomAccessibleInterval<FloatType> actual) {
		assertImagesEqual(new ImagePlus("expected", expected), actual);
	}

	private static <A extends Type<A>> String positionString(Cursor<Pair<A, A>> cursor) {
		StringJoiner joiner = new StringJoiner(", ");
		for (int i = 0, n = cursor.numDimensions(); i < n; i++)
			joiner.add(String.valueOf(cursor.getIntPosition(i)));
		return "(" + joiner + ")";
	}

	public static <T extends NumericType<T>> void showDifference(RandomAccessibleInterval<T> expectedImage, RandomAccessibleInterval<T> resultImage) {
		showDifference(Views.iterable(expectedImage), Views.iterable(resultImage));
	}

	public static <T extends NumericType<T>> void showDifference(IterableInterval<T> expectedImage, IterableInterval<T> resultImage) {
		ImageJ imageJ = new ImageJ();
		IterableInterval<T> difference = imageJ.op().math().subtract(expectedImage, resultImage);
		imageJ.ui().showUI();
		imageJ.ui().show(difference);
	}

	public static IterableInterval<IntType> loadImageIntType(String s) {
		IterableInterval<UnsignedByteType> img = ImagePlusAdapter.wrapByte(loadImage(s));
		return Converters.convert(img, (b, i) -> i.set(b.get()), new IntType());
	}

	public static RandomAccessibleInterval<FloatType> loadImageFloatType(String s) {
		return ImagePlusAdapter.wrapFloat(loadImage(s));
	}

	public static ImagePlus loadImage(String s) {
		return Utils.loadImagePlusFromResource(s);
	}

	public static float psnr(RandomAccessibleInterval<FloatType> expected, RandomAccessibleInterval<FloatType> actual) {
		return (float) (20 * Math.log10(max(expected)) - 10 * Math.log10(meanSquareError(expected, actual)));
	}

	private static float meanSquareError(RandomAccessibleInterval<FloatType> a, RandomAccessibleInterval<FloatType> b) {
		if(!Intervals.equals(a, b))
			throw new IllegalArgumentException("both arguments must be the same interval" +
					"given: " + showInterval(a) + " and: " + showInterval(b));
		DoubleType sum = new DoubleType(0.0f);
		Views.interval(Views.pair(a, b), a).forEach(x -> sum.set(sum.get() + sqr(x.getA().get() - x.getB().get())));
		return (float) (sum.get() / Intervals.numElements(a));
	}

	private static String showInterval(Interval b) {
		StringJoiner j = new StringJoiner(", ");
		int n = b.numDimensions();
		for (int i = 0; i < n; i++) j.add(b.min(i) + " - " + b.max(i));
		return "[" + j + "]";
	}

	private static float sqr(float v) {
		return v * v;
	}

	private static float max(RandomAccessibleInterval<FloatType> a) {
		IntervalView<FloatType> interval = Views.interval(a, a);
		FloatType result = interval.firstElement();
		interval.forEach(x -> result.set(Math.max(result.get(), x.get())));
		return result.get();
	}

	public static ImagePlus createImage(final String title, final int width, final int height, final int... pixels)
	{
		assertEquals( pixels.length, width * height );
		final byte[] bytes = new byte[pixels.length];
		for (int i = 0; i < bytes.length; i++) bytes[i] = (byte)pixels[i];
		final ByteProcessor bp = new ByteProcessor( width, height, bytes, null );
		return new ImagePlus( title, bp );
	}


	public static ImagePlus createImage(String title, int width, int height, final float... pixels) {
		assertEquals( pixels.length, width * height);
		final FloatProcessor processor = new FloatProcessor(width, height, pixels.clone());
		return new ImagePlus( title, processor);
	}

	public static void main(String... args) {
		new FeatureStackTest().testLegacyLipschitz();
	}
}
