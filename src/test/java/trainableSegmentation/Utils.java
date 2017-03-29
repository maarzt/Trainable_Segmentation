package trainableSegmentation;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.Unsigned2BitType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.util.Intervals;

import java.net.URL;
import java.util.NoSuchElementException;

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
		// create a cursor that automatically localizes itself on every move
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

	public static void showDifference(IterableInterval<IntType> resultImage, IterableInterval<IntType> expectedImage) {
		ImageJ imageJ = new ImageJ();
		IterableInterval<IntType> difference = imageJ.op().copy().iterableInterval(expectedImage);
		imageJ.op().math().subtract(difference, expectedImage, resultImage);
		imageJ.ui().showUI();
		imageJ.ui().show(difference);
	}

}
