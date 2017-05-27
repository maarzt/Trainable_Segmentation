package trainableSegmentation.filters;

/************************************************************************************\
 * 2D Lipschitz Filtering Plug-in for ImageJ                                          *
 * version 0.1, 2006/01/12  * written by Mikulas Stencel (mikulas.stencel@gmail.com)  *                                                     
 *        and  Jiri Janacek                                                           *
 * Adapted in 2010 by Ignacio Arganda-Carreras to be used as a library                *
 * This plug-in is designed to perform Filtering on an 8-bit, 16-bit                  *
 * and RGB images with support for ROI and Stacks. Long processing can be             *
 * stopped with Esc.                                                                  *
\************************************************************************************/

/**
*
* License: GPL
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License 2
* as published by the Free Software Foundation.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
* 
*/


/* importing standard Java API Files and ImageJ packages */

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.plugin.filter.PlugInFilter;
import ij.process.Blitter;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import java.awt.Rectangle;
import java.util.Arrays;
import java.util.Date;

public class Lipschitz_ implements PlugInFilter 
{
	private static final String Lipschitz_Version = "0.1";
	private static final String Lipschitz_Date = "2006/02/03";
	// the following are the input parameters, with default values assigned to them
	private boolean m_Down    = true;     //  
	private boolean m_TopHat    = false;  // lower Lipschitz cover  
	public double m_Slope = 10;                 // slope

	protected ImagePlus m_imp;
	private int m_scount;                      // number of stacks
	protected ImageStack m_stack, m_stack_out;

	protected Rectangle m_roi;
	private int ImageHeight = -1;
	private int ImageWidth = -1;
	private int m_channels = 0;
	boolean m_short = false;
	boolean breaked = false;
	private ImageProcessor iptmp;
	private int [] pixel;

	//-----------------------------------------------------------------------------------
	
	public void setDownHat(boolean downHat)
	{
		this.m_Down = downHat;
	}
	
	//-----------------------------------------------------------------------------------
	
	public void setTopHat(boolean topHat)
	{
		this.m_TopHat = topHat;
	}
	
	//-----------------------------------------------------------------------------------
	
	public int setup(String arg, ImagePlus imp) 
	{
		if (arg.equals("about")) 
		{
			showAbout();
			return DONE;
		}
		this.m_imp = imp;
		IJ.resetEscape();
		return DOES_8G + DOES_16+ DOES_RGB + NO_UNDO + NO_CHANGES;

	} // end of 'setup' method

	//-----------------------------------------------------------------------------------

	void showAbout() 
	{
		IJ.showMessage("About 2D Lipschitz filter",
				"version "+Lipschitz_Version+" ("+Lipschitz_Date+")\n"+
				"Mikulas Stencel, Jiri Janacek GPL2\n"+
				"Based on modul to Ellipse http://www.ellipse.sk/\n\n"+
				"This plugin is designed to filter images\n"+
		"using 2D Lipschitz filter.");
	} // end of 'showAbout' method

	//-----------------------------------------------------------------------------------

	private boolean GUI() 
	{
		GenericDialog gd = new GenericDialog("Lipschitz filter v"+Lipschitz_Version);
		gd.addNumericField("Slope", m_Slope, 2);
		String[] labels = {"TopDown", "TopHat"};
		boolean[] values = {m_Down, m_TopHat};
		gd.addCheckboxGroup(1, 2, labels, values);	
		gd.addMessage("Incorrect values will be replaced by defaults.\nLabels are drawn in the foreground color.\nPress Esc to stop processing.");
		return getUserParams(gd);
	}// end of 'GUI' method

	//-----------------------------------------------------------------------------------

	private boolean getUserParams(GenericDialog gd) 
	{
		gd.showDialog();
		// the user presses the Cancel button
		if (gd.wasCanceled()) return false;

		m_Slope = (double) gd.getNextNumber();
		if (m_Slope<=0) m_Slope = 10;

		m_Down = (boolean) gd.getNextBoolean();
		m_TopHat = (boolean) gd.getNextBoolean();
		return true;
	} // end of 'getUserParams' method

	//-----------------------------------------------------------------------------------

	public void run(ImageProcessor ip) 
	{
		m_stack = m_imp.getStack();
		m_scount = m_stack.getSize();
		
		if (GUI()) runLipschitz(ip);
	} // end of 'run' method

	//-----------------------------------------------------------------------------------
	public void Lipschitz2D(ImageProcessor ip)
	{     
		m_roi = ip.getRoi();
		ImageHeight = ip.getHeight();
		ImageWidth = ip.getWidth();
		m_channels = ip instanceof ColorProcessor ? 3 : 1;
		m_short = ip instanceof ShortProcessor;
		pixel = new int[m_channels];

		Progress progress = new Progress(2 * m_roi.height * m_channels);

		int sign = (m_Down ? 1 : -1 );

		int[][] srcPixels = copyAndMap1(ip, sign);

		int [][] destPixels = new int[m_channels][];
		for (int ii=0; ii < m_channels; ii++)
		{
			destPixels[ii] = srcPixels[ii].clone();
		}

		for (int z = 0; z < m_channels; z++)
			process1(progress, destPixels[z], srcPixels[z]);

		for (int z= 0; z < m_channels; z++)
			process2(progress, destPixels[z]);

		copyAndMap2(ip, destPixels, srcPixels, sign);
	}

	private int[][] copyAndMap1(ImageProcessor ip, int sign) {
		byte [][] tmpBytePixels = null;
		short [][] tmpShortPixels = null;
		int [][] srcPixels = new int[m_channels][ImageHeight * ImageWidth];

		if (m_channels == 1)
		{
			if (m_short)
			{
				tmpShortPixels = new short[m_channels][];
				tmpShortPixels[0] = (short []) ip.getPixels();
			}
			else
			{
				tmpBytePixels = new byte[m_channels][];
				tmpBytePixels[0] = (byte []) ip.getPixels();
			}

		}
		else
		{
			ColorProcessor cip = (ColorProcessor) ip;
			tmpBytePixels = new byte[m_channels][ImageHeight * ImageWidth];
			cip.getRGB(tmpBytePixels[0], tmpBytePixels[1], tmpBytePixels[2]);
		}

		for (int ii=0; ii < m_channels; ii++)
		{
			for (int ij=0; ij< ImageHeight * ImageWidth; ij++)
			{
				srcPixels[ii][ij] = (m_short? sign *(tmpShortPixels[ii][ij] & 0xffff):sign *(tmpBytePixels[ii][ij] & 0xff));
			}
		}

		return srcPixels;
	}

	private void process1(Progress progress, int[] destPixel, int[] srcPixel) {
		int sign = (m_Down ? 1 : -1 );
		int topdown = (m_Down ? 0 : 255);
		int slope = (int) (m_Slope);
		int slope1 = (int) (slope * Math.sqrt(2.0));
		int p2;
		int p3;
		int p;
		int p1;
		int p4;
		for (int y = m_roi.y; y < m_roi.y + m_roi.height; y++)   // rows
		{
			progress.step();
			p2= sign * (topdown + (sign) * slope);
			p3= sign * (topdown + (sign) * slope1);
			for (int x = m_roi.x; x < m_roi.x+m_roi.width; x++) // columns
			{
				p = (p2 - slope);
				p1 = (p3 - slope1);
				if (p1 > p) p= p1;
				p3 = destPixel[x + ImageWidth * (Math.max(y - 1,0))];
				p1 = p3 - slope;
				if (p1 > p) p= p1;

				p4 = destPixel[Math.min(x+1,ImageWidth-1) + ImageWidth * (Math.max(y - 1,0))] ;
				p1 = p4 - slope1;
				if (p1 > p) p= p1;

				p2 = srcPixel[x + ImageWidth * y];
				if (p > p2) {
					destPixel[x + ImageWidth * y] = p ;
					p2 = p;
				}
			}
		}
	}

	private void process2(Progress progress, int[] destPixel) {
		int sign = (m_Down ? 1 : -1 );
		int topdown = (m_Down ? 0 : 255);
		int slope = (int) (m_Slope);
		int slope1 = (int) (slope * Math.sqrt(2.0));
		int p2;
		int p3;
		int p;
		int p1;
		int p4;
		for (int y = m_roi.y+ m_roi.height - 1; y >= m_roi.y; y--)   // rows
		{
			progress.step();
			p2= sign * (topdown + (sign) * slope);
			p3= sign * (topdown + (sign) * slope1);
			for (int x= m_roi.x + m_roi.width - 1; x >= m_roi.x; x--)  // columns
			{
				p= (p2 - slope);
				p1= (p3 - slope1);
				if (p1 > p) p= p1;

				p3 = destPixel[x + ImageWidth * (Math.min(y + 1,ImageHeight-1))];
				p1 = p3 - slope;
				if (p1 > p)   p= p1;

				p4 = destPixel[Math.max(x-1,0) + ImageWidth * (Math.min(y + 1,ImageHeight-1))];
				p1 = p4 - slope1;
				if (p1 > p) p= p1;

				p2 = destPixel[x + ImageWidth * y];
				if (p > p2)
				{
					destPixel[x + ImageWidth * y] = p ;
					p2 = p;
				}
			}
		}
	}

	private void copyAndMap2(ImageProcessor ip, int[][] destPixels, int[][] srcPixels, int sign) {
		byte[][] tmpBytePixels = new byte[m_channels][ImageHeight * ImageWidth];
		short[][] tmpShortPixels = new short[m_channels][ImageHeight * ImageWidth];
		for (int ii=0; ii < m_channels; ii++)
		{
			for (int ij=0; ij< ImageHeight * ImageWidth; ij++)
			{
				if (m_TopHat)
				{
					tmpBytePixels[ii][ij]= (m_Down ? (byte) (srcPixels[ii][ij] - destPixels[ii][ij] + 255)
							: (byte) (destPixels[ii][ij] -srcPixels[ii][ij]));
				}
				else
				{
					if (m_short)
					{
						tmpShortPixels[ii][ij]= (short) ((sign * destPixels[ii][ij] & 0xffff));
					}
					else
					{
						tmpBytePixels[ii][ij]= (byte) (sign * destPixels[ii][ij]);
					}
				}
			}
		}

		if (m_channels == 1)
		{
			if (m_short)
			{
				ShortProcessor sip = (ShortProcessor) ip;
				sip.setPixels(tmpShortPixels[0]);
			}
			else
			{
				ByteProcessor bip = (ByteProcessor) ip;
				bip.setPixels(tmpBytePixels[0]);
			}

		}
		else
		{
			ColorProcessor cip = (ColorProcessor) ip;
			cip.setRGB(tmpBytePixels[0],tmpBytePixels[1],tmpBytePixels[2]);
		}
	}

	class Progress {
		private int step = 0;
		private final int max;

		Progress(int max) {
			this.max = max;
		}

		public void step() {
			IJ.showProgress(step++);
		}
	}


	public void runLipschitz(ImageProcessor ip) 
	{
		if (IJ.escapePressed()) return;
		breaked = false;
		Date d1, d2;
		d1 = new Date();

		IJ.showStatus("Initializing...");
		m_stack_out = m_imp.createEmptyStack(); 
		ImagePlus imp2 = null; 

		for(int i = 0; ((i < m_scount) && (!breaked)); i++)
		{
			if (m_scount >1)
			{
				ip = m_stack.getProcessor(i+1);
			}
			iptmp = ip.createProcessor(ImageWidth, ImageHeight);
			iptmp.copyBits(ip, 0, 0, Blitter.COPY);

			IJ.showStatus("Filtering "+ (i+1)+ "/"+m_scount +" slice.");


			Lipschitz2D(iptmp);


			m_stack_out.addSlice(m_imp.getShortTitle()+" "+(i+1)+"/"+m_scount, iptmp);

			if (breaked = IJ.escapePressed()) IJ.beep();
		}


		imp2 = new ImagePlus(m_imp.getShortTitle()+" Filtered (Lipschitz) Slope:"+m_Slope+" "+((m_Down)?" -Down":" ")+" "+((m_TopHat)?" -TopHat":" ")+((breaked)?" -INTERUPTED":""), m_stack_out);
		imp2.show();
		imp2.updateAndDraw();
		IJ.showProgress(1.0);


	} // end of 'runLipschitz' method

} // end of filter Class 
