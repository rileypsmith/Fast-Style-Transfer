import React from 'react';

import loading from './loading.gif';

const BASE_MEDIA_URL = "http://127.0.0.1:8000";

export default function Image({ image_response }) {
    return (
        <img src={BASE_MEDIA_URL + image_response.image} />
    )
};

export function NoImage() {
    return (
        <div className="no-image">
            <div className="d-flex justify-content-center align-items-center">
                <h3>No image selected.</h3>
            </div>
        </div>
    )
}

export function Loading() {
    return (
        <div className="no-image">
            <div className="d-flex justify-content-center align-items-center">
                <img src={loading} />
            </div>
        </div>
    )
}
