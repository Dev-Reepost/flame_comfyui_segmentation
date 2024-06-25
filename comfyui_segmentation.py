##########################################################################
#
# Filename: comfyui_segmentation.py
#
# Author: Julien Martin
# Created: 2024
#
###########################################################################

from __future__ import print_function

import sys
import json
import webbrowser
from enum import Enum
from pathlib import Path
from pprint import pprint

import pybox_v1 as pybox
import pybox_comfyui

from comfyui_client import COMFYUI_WORKING_DIR
from comfyui_client import COMFYUI_WORKFLOWS_DIR
from comfyui_client import find_models

from pybox_comfyui import DEFAULT_IMAGE_HEIGHT
from pybox_comfyui import IMAGE_HEIGHT_MAX
from pybox_comfyui import UI_INCVER
from pybox_comfyui import UI_INTERRUPT
from pybox_comfyui import UI_PROMPT
from pybox_comfyui import Color
from pybox_comfyui import LayerIn
from pybox_comfyui import LayerOut
from pybox_comfyui import PromptSign


COMFYUI_WORKFLOW_NAME = "ComfyUI SAM"
COMFYUI_OPERATOR_NAME = "segmentation"
COMFYUI_WORKFLOW_DIR = str(Path(COMFYUI_WORKFLOWS_DIR) / COMFYUI_OPERATOR_NAME / "api")
COMFYUI_WORKFLOW_FILENAME = "comfyui_segmentation_workflow_api.json"
COMFYUI_WORKFLOW_PATH = str(Path(COMFYUI_WORKFLOW_DIR) / COMFYUI_WORKFLOW_FILENAME)

COMFYUI_MODELS_SAM_DIR_PATHS = [
    str(Path(COMFYUI_WORKING_DIR) / "models" / "sams")
    ]

COMFYUI_MODELS_DINO_DIR_PATHS = [
    str(Path(COMFYUI_WORKING_DIR) / "models" / "groundingdino")
    ]

MODELS_NAMES = {
    "sams": {
        "sam_vit_h_4b8939.pth": "sam_vit_h (2.56GB)"
    },
    "dinos" : {
        "groundingdino_swint_ogc.pth" : "GroundingDINO_SwinT_OGC (694MB)"
    }
}

UI_MODELS_SAM_LIST = "SAM Model"
UI_MODELS_DINO_LIST = "DINO Model"
UI_THRESHOLD = "Threshold"
UI_SEGMENTRES = "Height"

DEFAULT_THRESHOLD = 0.3


class ComfyUISAM(pybox_comfyui.ComfyUIBaseClass):
    workflow_sam_matte_model_idx = -1
    workflow_dino_model_idx = -1
    workflow_sam_matte_idx = -1
    workflow_sam_segmentor_idx = -1
    
    operator_name = COMFYUI_OPERATOR_NAME
    operator_layers = [LayerIn.FRONT, LayerOut.RESULT, LayerOut.OUTMATTE]
    
    models_sam = []
    models_dino = []
    model_sam = ""
    model_dino = ""
    prompt = ""
    threshold = ""
    samsegmentor_res = ""
    
    init_cycle = True
    
    
    ###########################################################################
    # Overrided functions from pybox_comfyui.ComfyUIBaseClass
    
    
    def initialize(self):
        super().initialize()
        
        self.set_state_id("setup_ui")
        self.setup_ui()


    def setup_ui(self):
        super().setup_ui()
        
        self.set_state_id("execute")
    
    
    def execute(self):
        super().execute()
        
        if self.is_processing():
            if self.out_socket_active():
                self.submit_workflow()
        
        if self.get_global_element_value(UI_INTERRUPT):
            self.interrupt_workflow()

        self.update_workflow_execution()
        self.update_outputs()
    
    
    def teardown(self):
        super().teardown()
    
    
    ###########################################################################
    # Node-specific functions
    
    ###################################
    # UI
    
    def init_ui(self):
        
        # ComfyUI pages
        pages = []
        page = pybox.create_page(
            COMFYUI_WORKFLOW_NAME, 
            "Server & Workflow", "Prompt & Parameters", "Action"
            )
        pages.append(page)
        self.set_ui_pages_array(pages)
        
        col = 0
        self.set_ui_host_info(col)
        self.set_ui_workflow_path(col, COMFYUI_WORKFLOW_DIR, COMFYUI_WORKFLOW_PATH)
        
        col = 1
        # ComfyUI Segment Anything prompt conditioning
        prompt = pybox.create_text_field(
            UI_PROMPT(PromptSign.POSITIVE, 0), 
            row=0, col=col, value=self.prompt
            )
        self.add_global_elements(prompt)
        
        #ComfyUI Segmentation threshold
        threshold = pybox.create_float_numeric(
            UI_THRESHOLD, 
            value=self.threshold, 
            default=DEFAULT_THRESHOLD, 
            min=0, max=1, inc=0.01,
            row=1, col=col, tooltip="Segmentation threshold",
            )
        self.add_global_elements(threshold)
        
        # ComfyUI Segmentation resolution
        segment_res = pybox.create_float_numeric(
            UI_SEGMENTRES, 
            value=int(self.get_resolution()["height"]), 
            default=DEFAULT_IMAGE_HEIGHT, 
            min=0, max=IMAGE_HEIGHT_MAX, inc=1,
            row=2, col=col, tooltip="Segmentation image height",
            )
        self.add_global_elements(segment_res)
        
        # ComfyUI Segmentation Anything models filename
        models_list = pybox.create_popup(
            UI_MODELS_SAM_LIST, 
            self.models_sam, 
            value=self.models_sam.index(self.model_sam), 
            default=0, 
            row=3, col=col, tooltip="Segment Anything model to use"
            )
        self.add_global_elements(models_list)
        
        # ComfyUI Grounding DINO models filename
        models_list = pybox.create_popup(
            UI_MODELS_DINO_LIST, 
            self.models_dino, 
            value=self.models_dino.index(self.model_dino), 
            default=0, 
            row=4, col=col, tooltip="Grounding DINO model to use"
            )
        self.add_global_elements(models_list)
        
        col = 2
        # ComfyUI workflow actions
        self.ui_version_row = 0
        self.ui_version_col = col
        self.set_ui_versions(row=0, col=col)
        
        self.set_ui_increment_version(col)
        
        self.set_ui_interrupt(col)
        
        self.ui_processing_color_row = 3
        self.ui_processing_color_col = col
        self.set_ui_processing_color(Color.GRAY, self.ui_processing)
    
    
    ###################################
    # Helpers 
    
    def set_models(self):
        sam_models = find_models(COMFYUI_MODELS_SAM_DIR_PATHS)
        self.models_sam = [MODELS_NAMES["sams"][model_fn] for model_fn in sam_models]
        dino_models = find_models(COMFYUI_MODELS_DINO_DIR_PATHS)
        self.models_dino = [MODELS_NAMES["dinos"][model_fn] for model_fn in dino_models]
    
    
    ###################################
    # Workflow
    
    def load_workflow(self):
        with open(COMFYUI_WORKFLOW_PATH) as f:
            print("Loading Workflow")
            self.workflow = json.load(f)
            self.workflow_id_to_class_type = {id: details['class_type'] for id, details in self.workflow.items()}
            self.workflow_load_exr_front_idx = self.get_workflow_index('LoadEXR')
            save_exr_nodes = [(key, self.workflow.get(key)["inputs"]) for key, value in self.workflow_id_to_class_type.items() if value == 'SaveEXR']
            self.workflow_save_exr_outmatte_idx = [key for (key, attr) in save_exr_nodes if attr["filename_prefix"] == "OutMatte"][0]
            self.workflow_save_exr_result_idx = [key for (key, attr) in save_exr_nodes if attr["filename_prefix"] == "Result"][0]
            self.workflow_sam_matte_idx = self.get_workflow_index('GroundingDinoSAMSegment (segment anything)')
            self.workflow_sam_segmentor_idx = self.get_workflow_index('SAMPreprocessor')
            self.workflow_sam_matte_model_idx = self.workflow.get(self.workflow_sam_matte_idx)["inputs"]["sam_model"][0]
            self.workflow_dino_model_idx = self.workflow.get(self.workflow_sam_matte_idx)["inputs"]["grounding_dino_model"][0]
            self.model_sam = self.workflow.get(self.workflow_sam_matte_model_idx)["inputs"]["model_name"]
            self.model_dino = self.workflow.get(self.workflow_dino_model_idx)["inputs"]["model_name"]
            self.prompt = self.workflow.get(self.workflow_sam_matte_idx)["inputs"]["prompt"]
            self.threshold = self.workflow.get(self.workflow_sam_matte_idx)["inputs"]["threshold"]
            self.out_frame_pad = self.workflow.get(self.workflow_save_exr_outmatte_idx)["inputs"]["frame_pad"]
    
    
    def set_workflow_models(self):
        sam_model_idx = self.get_global_element_value(UI_MODELS_SAM_LIST)
        self.model_sam = self.models_sam[sam_model_idx]
        self.workflow.get(self.workflow_sam_matte_model_idx)["inputs"]["model_name"] = self.model_sam
        print(f'Workflow SAM model: {self.model_sam}')
        dino_model_idx = self.get_global_element_value(UI_MODELS_DINO_LIST)
        self.model_dino = self.models_dino[dino_model_idx]
        self.workflow.get(self.workflow_dino_model_idx)["inputs"]["model_name"] = self.model_dino
        print(f'Workflow DINO model: {self.model_dino}')
    
    
    def set_workflow_prompt(self):
        if self.workflow:
            prompt_name_pos = UI_PROMPT(PromptSign.POSITIVE, 0)
            self.prompt = self.get_global_element_value(prompt_name_pos).strip()
            self.workflow.get(self.workflow_sam_matte_idx)["inputs"]["prompt"] = self.prompt
            print(f'Workflow Prompt: {self.prompt}')
    
    
    def set_workflow_resolution(self):
        if self.workflow:  
            self.samsegmentor_res = int(self.get_global_element_value(UI_SEGMENTRES))
            self.workflow.get(self.workflow_sam_segmentor_idx)["inputs"]["resolution"] = self.samsegmentor_res
            print(f'Workflow Segmentation resolution: {self.samsegmentor_res}')
    
    
    def set_workflow_threshold(self):
        if self.workflow:  
            self.threshold = round(self.get_global_element_value(UI_THRESHOLD), 2)
            self.workflow.get(self.workflow_sam_matte_idx)["inputs"]["threshold"] = self.threshold
            print(f'Workflow Matting threshold: {self.threshold}')
    
    
    def workflow_setup(self):
        self.set_workflow_models()
        self.set_workflow_prompt()
        self.set_workflow_threshold()
        self.set_workflow_resolution()
        self.set_workflow_load_exr_filepath()
        self.set_workflow_save_exr_filename_prefix()
    
    
def _main(argv):
    print("____________________________________________________________")
    print("Loading JSON Pybox")
    print("____________________________________________________________")
    
    # Load the json file, make sure you have read access to it
    p = ComfyUISAM(argv[0])
    # Call the appropriate function
    p.dispatch()
    # Save file
    p.write_to_disk(argv[0])
    
    print("____________________________________________________________")
    print("Wrote JSON Pybox")
    print("____________________________________________________________")

if __name__ == "__main__":
    _main(sys.argv[1:])
