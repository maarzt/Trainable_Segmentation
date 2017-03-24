package trainableSegmentation;

import hr.irb.fastRandomForest.FastRandomForest;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.process.ImageConverter;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.type.numeric.integer.IntType;

/**
 * @author Matthias Arzt
 */
public class Classifier {

	private final WekaSegmentationIJ2 wekaSegmentation;

	public Classifier(WekaSegmentationIJ2 wekaSegmentation) {
		this.wekaSegmentation = wekaSegmentation;
	}

	public ImagePlus apply(ImagePlus image) {
		ImagePlus output = wekaSegmentation.applyClassifier(image);
		new ImageConverter(output).convertToGray8();
		return output;
	}

	public static Classifier train(ImagePlus image, ImgLabeling<String, IntType> labeling) {
		WekaSegmentationIJ2 segmentator = new WekaSegmentationIJ2( image );
		segmentator.addExample( 0, new Roi( 10, 10, 50, 50 ), 1 );
		segmentator.addExample( 1, new Roi( 400, 400, 30, 30 ), 1 );

		FastRandomForest rf = (FastRandomForest) segmentator.getClassifier();
		rf.setSeed( 69 );
		if(! segmentator.trainClassifier(labeling) )
			throw new IllegalStateException();
		return new Classifier(segmentator);
	}

}
