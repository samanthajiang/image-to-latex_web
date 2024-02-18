import React, { useState } from 'react';

// import Prism from 'prismjs';
import { highlight, languages, Grammar } from 'prismjs';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import Editor from 'react-simple-code-editor';

import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-latex';
import 'prismjs/themes/prism.css';

function EditableCode({ value }: any) {
  const [code, setCode] = useState(value);
  // const makehighlight = () => {
  //   if (Prism.languages.latex !== undefined) {
  //     Prism.highlight(code, Prism.languages.latex, 'latex');
  //   }
  // };
  return (
    <div className="prediction">
      <Editor
        className="shadow-sm bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-primary-500 dark:focus:border-primary-500 dark:shadow-sm-light"
        value={code}
        padding={10}
        onValueChange={(code) => setCode(code)}
        // highlight={code => highlight(code, languages['js'], 'javascript')}
        highlight={(code) =>
          highlight(code, languages.latex as Grammar, 'latex')
        }
        // highlight={code => makehighlight(code)}
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
