import React, { useState } from 'react';

import { highlight, languages } from 'prismjs/components/prism-core';
import Editor from 'react-simple-code-editor';

import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/themes/prism.css';
import { BlockMath } from 'react-katex';
import LoadingSpinner from "./Spinner";

function EditableCode({ value }) {
  const [code, setCode] = useState(value);
  return (
      <div className="prediction">

          <Editor className='shadow-sm bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-primary-500 dark:focus:border-primary-500 dark:shadow-sm-light'
      value={code}
      padding={10}
      onValueChange={(code) => setCode(code)}
      highlight={(code) => highlight(code, languages.js)}
      style={{
        fontFamily: 'monospace',
        fontSize: 17,
      }}
    />

            <BlockMath>{code}</BlockMath>

    </div>
  );
}
export default EditableCode;
