package org.cloudbus.cloudsim.cost.models.CostModelAzure;

import org.cloudbus.cloudsim.cost.models.CostModel;

public class CostModelAzure extends CostModel {

    public CostModelAzure(
            Region region,
            OS os,
            Tier tier,
            Instance instance
    ){
        super(0, getAzureCost(region, os, tier, instance), 0);
    }

    /** Azure costs as of 20-June-2019. Source - https://azure.microsoft.com/en-au/pricing/calculator/ **/
    public static double getAzureCost(Region region, OS os, Tier tier, Instance instance){
        switch (region){
            case EastUS:
                switch (os){
                    case Linux:
                        switch (tier){
                            case Basic:
                                switch (instance){
                                    case A0: return 0.0247;
                                    case A1: return 0.0316;
                                    case A2: return 0.1085;
                                }
                            case Standard:
                                switch (instance){
                                    case A0: return 0.0275;
                                    case A1: return 0.0824;
                                    case A2: return 0.1648;
                                    case B1S: return 0.0143;
                                    case B2S: return 0.0571;
                                }
                            case Low_Priority:
                                switch (instance){
                                    case A1: return 0.0165;
                                    case A2: return 0.033;
                                }
                        }
                    case Windows:
                        switch (tier){
                            case Basic:
                                switch (instance){
                                    case A0: return 0.0247;
                                    case A1: return 0.0439;
                                    case A2: return 0.1826;
                                }
                            case Standard:
                                switch (instance){
                                    case A0: return 0.0275;
                                    case A1: return 0.1236;
                                    case A2: return 0.2471;
                                    case B1S: return 0.0192;
                                    case B2S: return 0.0681;
                                }
                            case Low_Priority:
                                switch (instance){
                                    case A1: return 0.0494;
                                    case A2: return 0.0989;
                                }
                        }
                }
            case Australia_SouthEast:
                switch (os){
                    case Linux:
                        switch (tier){
                            case Basic:
                                switch (instance){
                                    case A1: return 0.033;
                                    case A2: return 0.0439;
                                }
                            case Standard:
                                switch (instance){
                                    case A0: return 0.0398;
                                    case A1: return 0.0975;
                                    case A2: return 0.195;
                                    case B1S: return 0.0181;
                                    case B2S: return 0.0725;
                                }
                            case Low_Priority:
                                switch (instance){
                                    case A1: return 0.0192;
                                    case A2: return 0.0384;
                                }
                        }
                    case Windows:
                        switch (tier){
                            case Basic:
                                switch (instance){
                                    case A0: return 0.033;
                                    case A1: return 0.0563;
                                    case A2: return 0.2293;
                                }
                            case Standard:
                                switch (instance){
                                    case A0: return 0.0398;
                                    case A1: return 0.1551;
                                    case A2: return 0.3103;
                                    case B1S: return 0.0236;
                                    case B2S: return 0.0835;
                                    case D64: return 6.944;
                                    case D32: return 3.472;
                                    case B4ms: return 0.227;
                                    case B2ms: return  0.114;
                                }
                            case Low_Priority:
                                switch (instance){
                                    case A1: return 0.0618;
                                    case A2: return 0.1236;
                                }
                        }
                }
        }
        return 0;
    }
}
