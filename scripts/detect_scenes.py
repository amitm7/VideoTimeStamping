from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def get_scene_timestamps(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
