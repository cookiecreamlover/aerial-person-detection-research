from evaluation.simulate_uav_pipeline import run


def test_simulated_pipeline_labels_frames_and_candidates():
    records = [
        {'image_id': 'a', 'predictions': [{'confidence': .9, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10, 'validation_selected_threshold': .5}]},
        {'image_id': 'b', 'predictions': [{'confidence': .8, 'x1': 2, 'y1': 1, 'x2': 12, 'y2': 11, 'validation_selected_threshold': .5}]},
    ]
    output = run(records)
    assert output[0]['pipeline_status'] == 'simulated_uav_frame_processed'
    assert output[0]['candidate_persons'][0]['candidate_id'] == output[1]['candidate_persons'][0]['candidate_id']


if __name__ == '__main__':
    test_simulated_pipeline_labels_frames_and_candidates()
    print('UAV simulation test passed.')
